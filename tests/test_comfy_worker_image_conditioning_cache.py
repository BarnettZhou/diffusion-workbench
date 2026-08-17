"""图片解码缓存与 H3 图片条件缓存的无 GPU 单元测试。

使用 fake node/fake tensor，不导入真实 ComfyUI 模型；断言以节点调用次数为准。
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import diffusion_workbench_core.comfy_worker as comfy_worker_module
from diffusion_workbench_core.comfy_worker import ComfyWorker, _LruTensorCache


class _FakeTensor:
    """duck-typed 张量替身：只提供缓存逻辑需要的接口，不做真实计算。"""

    def __init__(self, data: bytes = b"", shape=(1, 2, 2, 3)):
        self._data = data
        self.shape = shape

    def nelement(self):
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def element_size(self):
        return 4

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self._data

    def clone(self):
        return _FakeTensor(self._data, self.shape)


class _FakeH3ImageToVideo:
    calls = 0

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length, first_frame=None, last_frame=None):
        cls.calls += 1
        conditioning = [
            [
                _FakeTensor(b"fl2va-cond", (1, 4, 8)),
                {
                    "minimax_keyframes": [{"latent": _FakeTensor(b"kf", (1, 4, 4, 4))}],
                    "minimax_frame_count": 33,
                },
            ]
        ]
        return conditioning, {"samples": "fresh"}


class _FakeH3ReferenceToVideo:
    calls = 0

    @classmethod
    def execute(
        cls, clip, vae, audio_vae, prompt, width, height, length,
        ref_image_size="match", ref_images=None,
    ):
        cls.calls += 1
        conditioning = [
            [
                _FakeTensor(b"ref2va-cond", (1, 4, 8)),
                {"minimax_refs": [{"latent": _FakeTensor(b"ref", (1, 4, 4, 4))}]},
            ]
        ]
        return conditioning, {"samples": "fresh"}


class _FakeEmptyLatent:
    calls = 0

    @classmethod
    def execute(cls, width, height, length):
        cls.calls += 1
        return ({"samples": f"latent-{cls.calls}"},)


class _FakeSigmaShift:
    @classmethod
    def execute(cls, model, shift, audio_shift):
        return ("sampling-model",)


class _FakeVAEDecode:
    def decode(self, vae, samples):
        return ("images",)


class _FakeVAEDecodeAudio:
    @staticmethod
    def execute(audio_vae, samples):
        return ("audio",)


class _FakeCuda:
    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return 0

    def max_memory_reserved(self):
        return 0


class _FakeTorch:
    cuda = _FakeCuda()


class _FakeFingerprints:
    @staticmethod
    def describe(path):
        return {"filename": Path(path).name}


def _make_worker() -> ComfyWorker:
    worker = object.__new__(ComfyWorker)
    worker._conditioning_cache = {}
    worker.model_path = Path("model.safetensors")
    worker.vae_path = Path("vae.safetensors")
    worker.audio_vae_path = Path("audio_vae.safetensors")
    worker.clip_path = Path("clip.safetensors")
    worker.clip_type = "minimax"
    worker.clip = object()
    worker.vae = object()
    worker.audio_vae = object()
    worker.MiniMaxH3ImageToVideo = _FakeH3ImageToVideo
    worker.MiniMaxH3ReferenceToVideo = _FakeH3ReferenceToVideo
    return worker


def _command(**overrides) -> dict:
    command = {
        "prompt": "一只猫在花园里",
        "width": 1344,
        "height": 768,
        "length": 124,
    }
    command.update(overrides)
    return command


def _h3_command(output_dir: Path, **overrides) -> dict:
    command = {
        "job_id": "job-1",
        "video_model": "minimax_h3",
        "model_path": "model.safetensors",
        "vae_path": "vae.safetensors",
        "audio_vae_path": "audio_vae.safetensors",
        "text_encoder_path": "clip.safetensors",
        "clip_type": "minimax",
        "prompt": "一只猫在花园里",
        "negative_prompt": "",
        "width": 1344,
        "height": 768,
        "length": 124,
        "duration_seconds": 5,
        "fps": 24,
        "steps": 8,
        "seed": 42,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "shift": 12.0,
        "audio_shift": 3.0,
        "latent_multiplier": 1.0,
        "input_image_path": "frame.png",
        "reference_image_path": None,
        "output_path": str(output_dir / "out.mp4"),
        "workbench_version": "0.1.0",
    }
    command.update(overrides)
    return command


def _make_h3_branch_worker() -> ComfyWorker:
    """构造可运行 _generate_minimax_h3 全分支的 fake worker。"""

    worker = _make_worker()
    worker.mode = None
    worker.model = object()
    worker.clip = object()
    worker.vae = object()
    worker.audio_vae = object()
    worker._ensure_model = lambda path: False
    worker._ensure_clip = lambda path, clip_type: False
    worker._ensure_vae = lambda path: False
    worker._ensure_audio_vae = lambda path: False
    worker.resource_fingerprints = _FakeFingerprints()
    worker.runtime_versions = {}
    worker.torch = _FakeTorch()
    worker.MiniMaxH3SigmaShift = _FakeSigmaShift
    worker.EmptyMiniMaxH3LatentAV = _FakeEmptyLatent
    worker.nodes = type("Nodes", (), {"VAEDecode": _FakeVAEDecode})
    worker.VAEDecodeAudio = _FakeVAEDecodeAudio
    worker.sample_calls = 0

    def fake_sample(*args, **kwargs):
        worker.sample_calls += 1
        return {"samples": "sampled"}

    worker._sample = fake_sample
    worker._save_video = lambda *args, **kwargs: None
    worker._load_input_image_cached = (
        lambda path: (_FakeTensor(), ("fp", str(path))) if path else (None, None)
    )
    return worker


class LruTensorCacheTests(unittest.TestCase):
    def test_hit_updates_lru_order(self):
        cache = _LruTensorCache(max_items=2, max_bytes=1024)
        cache.put("a", "va", 10)
        cache.put("b", "vb", 10)
        self.assertEqual(cache.get("a"), "va")
        cache.put("c", "vc", 10)
        # "a" 被命中过，最久未使用的 "b" 被淘汰。
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "va")
        self.assertEqual(cache.get("c"), "vc")

    def test_evicts_by_total_bytes(self):
        cache = _LruTensorCache(max_items=8, max_bytes=100)
        cache.put("a", "va", 60)
        cache.put("b", "vb", 60)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), "vb")

    def test_entry_over_budget_is_not_cached(self):
        cache = _LruTensorCache(max_items=4, max_bytes=100)
        self.assertFalse(cache.put("big", "vb", 101))
        self.assertIsNone(cache.get("big"))

    def test_clear(self):
        cache = _LruTensorCache(max_items=4, max_bytes=100)
        cache.put("a", "va", 10)
        cache.clear()
        self.assertIsNone(cache.get("a"))
        self.assertTrue(cache.put("b", "vb", 100))


class H3ImageConditioningCacheTests(unittest.TestCase):
    def setUp(self):
        _FakeH3ImageToVideo.calls = 0
        _FakeH3ReferenceToVideo.calls = 0
        _FakeEmptyLatent.calls = 0

    def test_same_fl2va_input_hits_cache_and_returns_clones(self):
        worker = _make_worker()
        command = _command()
        fingerprint = ((1, 2, 2, 3), "digest")

        first = worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint)
        second = worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint)

        self.assertEqual(_FakeH3ImageToVideo.calls, 1)
        # 命中返回深拷贝，缓存对象不直接交给采样器。
        self.assertIsNot(first, second)
        self.assertIsNot(first[0][0], second[0][0])

    def test_same_ref2va_input_hits_cache(self):
        worker = _make_worker()
        command = _command()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_ref2va_conditioning(command, _FakeTensor(), fingerprint)
        worker._get_h3_ref2va_conditioning(command, _FakeTensor(), fingerprint)

        self.assertEqual(_FakeH3ReferenceToVideo.calls, 1)

    def test_prompt_change_misses(self):
        worker = _make_worker()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_fl2va_conditioning(_command(prompt="a"), _FakeTensor(), fingerprint)
        worker._get_h3_fl2va_conditioning(_command(prompt="b"), _FakeTensor(), fingerprint)

        self.assertEqual(_FakeH3ImageToVideo.calls, 2)

    def test_image_content_change_misses(self):
        worker = _make_worker()
        command = _command()

        worker._get_h3_fl2va_conditioning(command, _FakeTensor(), ((1, 2, 2, 3), "digest-a"))
        worker._get_h3_fl2va_conditioning(command, _FakeTensor(), ((1, 2, 2, 3), "digest-b"))

        self.assertEqual(_FakeH3ImageToVideo.calls, 2)

    def test_size_change_misses(self):
        worker = _make_worker()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_fl2va_conditioning(_command(width=1344), _FakeTensor(), fingerprint)
        worker._get_h3_fl2va_conditioning(_command(width=1024), _FakeTensor(), fingerprint)
        worker._get_h3_ref2va_conditioning(_command(height=768), _FakeTensor(), fingerprint)
        worker._get_h3_ref2va_conditioning(_command(height=512), _FakeTensor(), fingerprint)

        self.assertEqual(_FakeH3ImageToVideo.calls, 2)
        self.assertEqual(_FakeH3ReferenceToVideo.calls, 2)

    def test_length_change_misses(self):
        worker = _make_worker()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_fl2va_conditioning(_command(length=124), _FakeTensor(), fingerprint)
        worker._get_h3_fl2va_conditioning(_command(length=209), _FakeTensor(), fingerprint)

        self.assertEqual(_FakeH3ImageToVideo.calls, 2)

    def test_ref_image_size_change_misses(self):
        worker = _make_worker()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_ref2va_conditioning(_command(), _FakeTensor(), fingerprint)
        worker._get_h3_ref2va_conditioning(
            _command(ref_image_size="max"), _FakeTensor(), fingerprint
        )

        self.assertEqual(_FakeH3ReferenceToVideo.calls, 2)

    def test_fl2va_and_ref2va_do_not_share_entries(self):
        worker = _make_worker()
        command = _command()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint)
        worker._get_h3_ref2va_conditioning(command, _FakeTensor(), fingerprint)

        self.assertEqual(_FakeH3ImageToVideo.calls, 1)
        self.assertEqual(_FakeH3ReferenceToVideo.calls, 1)

    def test_resource_identity_change_misses(self):
        worker = _make_worker()
        command = _command()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_ref2va_conditioning(command, _FakeTensor(), fingerprint)
        worker.vae_path = Path("vae-2.safetensors")
        worker._get_h3_ref2va_conditioning(command, _FakeTensor(), fingerprint)
        worker.audio_vae_path = Path("audio-vae-2.safetensors")
        worker._get_h3_ref2va_conditioning(command, _FakeTensor(), fingerprint)
        worker.clip_path = Path("clip-2.safetensors")
        worker._get_h3_ref2va_conditioning(command, _FakeTensor(), fingerprint)
        worker.model_path = Path("model-2.safetensors")
        worker._get_h3_ref2va_conditioning(command, _FakeTensor(), fingerprint)

        # 初始 1 次 + vae/audio_vae/text encoder/模型各变 1 次，全部未命中。
        self.assertEqual(_FakeH3ReferenceToVideo.calls, 5)

    def test_clear_conditioning_caches_invalidates(self):
        worker = _make_worker()
        command = _command()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint)
        worker._clear_conditioning_caches()
        worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint)

        self.assertEqual(_FakeH3ImageToVideo.calls, 2)

    def test_lru_capacity_evicts_least_recently_used(self):
        worker = _make_worker()
        fingerprint = ((1, 2, 2, 3), "digest")

        worker._get_h3_fl2va_conditioning(_command(prompt="a"), _FakeTensor(), fingerprint)
        worker._get_h3_fl2va_conditioning(_command(prompt="b"), _FakeTensor(), fingerprint)
        # 容量为 2，第三条目淘汰最久未使用的 "a"。
        worker._get_h3_fl2va_conditioning(_command(prompt="c"), _FakeTensor(), fingerprint)
        self.assertEqual(_FakeH3ImageToVideo.calls, 3)

        worker._get_h3_fl2va_conditioning(_command(prompt="b"), _FakeTensor(), fingerprint)
        self.assertEqual(_FakeH3ImageToVideo.calls, 3)

        worker._get_h3_fl2va_conditioning(_command(prompt="a"), _FakeTensor(), fingerprint)
        self.assertEqual(_FakeH3ImageToVideo.calls, 4)

    def test_entry_over_budget_is_not_cached_but_returned(self):
        worker = _make_worker()
        fingerprint = ((1, 2, 2, 3), "digest")
        # 伪造超大张量：201M 元素 × 4 字节 ≈ 768 MiB，超过 512 MiB 预算。
        huge = _FakeTensor(b"huge", (1, 3, 8192, 8192))

        class HugeNode:
            calls = 0

            @classmethod
            def execute(cls, clip, vae, prompt, width, height, length, first_frame=None, last_frame=None):
                cls.calls += 1
                return [[huge, {}]], {"samples": "fresh"}

        worker.MiniMaxH3ImageToVideo = HugeNode
        command = _command()

        first = worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint)
        second = worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(HugeNode.calls, 2)


class InputImageDecodeCacheTests(unittest.TestCase):
    def setUp(self):
        _FakeH3ImageToVideo.calls = 0
        _FakeH3ReferenceToVideo.calls = 0
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_decode_worker(self):
        worker = _make_worker()
        worker.uncached_calls = 0

        def fake_uncached(path):
            worker.uncached_calls += 1
            return _FakeTensor(data=Path(path).read_bytes())

        worker._load_input_image_uncached = fake_uncached
        return worker

    def test_empty_path_returns_none(self):
        worker = self._make_decode_worker()
        self.assertIsNone(worker._load_input_image(None))
        self.assertEqual(worker._load_input_image_cached(""), (None, None))

    def test_missing_file_raises_chinese_error(self):
        worker = self._make_decode_worker()
        with self.assertRaisesRegex(FileNotFoundError, "找不到输入图片"):
            worker._load_input_image(str(self.tmp_path / "missing.png"))

    def test_decode_failure_propagates(self):
        worker = _make_worker()

        def broken(path):
            raise RuntimeError("图片损坏")

        worker._load_input_image_uncached = broken
        image_path = self.tmp_path / "broken.png"
        image_path.write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "图片损坏"):
            worker._load_input_image(str(image_path))

    def test_same_file_decode_hits_cache(self):
        worker = self._make_decode_worker()
        image_path = self.tmp_path / "frame.png"
        image_path.write_bytes(b"first-content")

        tensor1, fingerprint1 = worker._load_input_image_cached(str(image_path))
        tensor2, fingerprint2 = worker._load_input_image_cached(str(image_path))

        self.assertEqual(worker.uncached_calls, 1)
        self.assertEqual(fingerprint1, fingerprint2)
        # 返回 clone，不共享缓存副本。
        self.assertIsNot(tensor1, tensor2)

    def test_content_change_produces_new_fingerprint_and_cache_miss(self):
        worker = self._make_decode_worker()
        image_path = self.tmp_path / "frame.png"
        image_path.write_bytes(b"first-content")
        command = _command()

        _, fingerprint1 = worker._load_input_image_cached(str(image_path))
        worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint1)

        image_path.write_bytes(b"second-content-longer")
        _, fingerprint2 = worker._load_input_image_cached(str(image_path))
        self.assertNotEqual(fingerprint1, fingerprint2)
        worker._get_h3_fl2va_conditioning(command, _FakeTensor(), fingerprint2)

        self.assertEqual(worker.uncached_calls, 2)
        self.assertEqual(_FakeH3ImageToVideo.calls, 2)


class MinimaxH3BranchTests(unittest.TestCase):
    """分支级测试：证明 _generate_minimax_h3 选择正确路径且 latent 每任务重建。"""

    def setUp(self):
        _FakeH3ImageToVideo.calls = 0
        _FakeH3ReferenceToVideo.calls = 0
        _FakeEmptyLatent.calls = 0
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        emit_patch = mock.patch.object(comfy_worker_module, "emit", lambda payload: None)
        self.addCleanup(emit_patch.stop)
        emit_patch.start()

    def tearDown(self):
        self._tmp.cleanup()

    def test_fl2va_branch_caches_conditioning_and_recreates_latent(self):
        worker = _make_h3_branch_worker()
        command = _h3_command(self.tmp_path)

        worker._generate_minimax_h3(command)
        worker._generate_minimax_h3(dict(command, job_id="job-2"))

        self.assertEqual(_FakeH3ImageToVideo.calls, 1)
        self.assertEqual(_FakeH3ReferenceToVideo.calls, 0)
        self.assertEqual(_FakeEmptyLatent.calls, 2)
        self.assertEqual(worker.sample_calls, 2)

    def test_ref2va_branch_caches_conditioning_and_recreates_latent(self):
        worker = _make_h3_branch_worker()
        command = _h3_command(
            self.tmp_path, input_image_path=None, reference_image_path="ref.png"
        )

        worker._generate_minimax_h3(command)
        worker._generate_minimax_h3(dict(command, job_id="job-2"))

        self.assertEqual(_FakeH3ReferenceToVideo.calls, 1)
        self.assertEqual(_FakeH3ImageToVideo.calls, 0)
        self.assertEqual(_FakeEmptyLatent.calls, 2)
        self.assertEqual(worker.sample_calls, 2)


if __name__ == "__main__":
    unittest.main()
