from __future__ import annotations

import asyncio
import itertools

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .schemas import public_event


class EventHub:
    """Core 后台线程事件到 asyncio 订阅者的单点桥接。

    Core 只允许一个 event sink；所有 WebSocket 连接共享本 Hub 扇出。
    慢客户端的队列满时丢弃最旧事件，绝不反向阻塞生成线程。
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._closed = False
        self._event_ids = itertools.count(1)

    def emit_from_core_thread(self, event: dict) -> None:
        if not self._closed:
            self._loop.call_soon_threadsafe(self._publish, event.copy())

    def _publish(self, event: dict) -> None:
        message = public_event(event, next(self._event_ids))
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(message)

    def subscribe(self, max_events: int = 128) -> asyncio.Queue[dict | None]:
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=max_events)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict | None]) -> None:
        self._subscribers.discard(queue)

    def close(self) -> None:
        self._closed = True
        # 给每个订阅者投递哨兵,唤醒阻塞在 queue.get() 的连接处理任务
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self._subscribers.clear()


router = APIRouter()


@router.websocket("/api/v1/events")
async def events(websocket: WebSocket):
    await websocket.accept()
    hub: EventHub = websocket.app.state.event_hub
    queue = hub.subscribe()
    try:
        while True:
            # 同时等待新事件与连接状态:只等 queue.get() 会导致连接被服务端
            # 关闭时处理任务无法退出(优雅关闭超时被强制 cancel)
            get_task = asyncio.ensure_future(queue.get())
            recv_task = asyncio.ensure_future(websocket.receive())
            done, _ = await asyncio.wait(
                {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                get_task.cancel()
                message = recv_task.result()  # 断开时抛 WebSocketDisconnect
                if message["type"] == "websocket.disconnect":
                    break
                continue  # 忽略客户端消息(前端目前不会发送)
            recv_task.cancel()
            event = get_task.result()
            if event is None:  # Hub 关闭哨兵
                break
            await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except asyncio.CancelledError:
        pass  # 服务优雅退出超时会取消等待中的连接,静默结束即可
    finally:
        hub.unsubscribe(queue)
