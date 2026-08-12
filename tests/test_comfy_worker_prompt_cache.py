import unittest

from diffusion_workbench_core.comfy_worker import ComfyWorker


class _FakeClip:
    def __init__(self):
        self.tokenize_calls = 0
        self.encode_calls = 0

    def tokenize(self, text, **kwargs):
        self.tokenize_calls += 1
        return (text, tuple(sorted(kwargs.items())))

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        return {"tokens": tokens, "encode_call": self.encode_calls}


class _FakeClipTextEncode:
    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens),)


class PromptConditioningCacheTests(unittest.TestCase):
    def test_reuses_positive_and_negative_conditioning_for_same_clip_and_prompt(self):
        worker = object.__new__(ComfyWorker)
        worker.clip = _FakeClip()
        worker.nodes = type("Nodes", (), {"CLIPTextEncode": _FakeClipTextEncode})
        worker.clip_path = "clip.safetensors"
        worker.clip_type = "stable_diffusion"

        first = worker._encode_text_conditioning(
            "portrait", "", cfg=1.0, cache_scope="image", reuse_negative_at_cfg_one=True
        )
        second = worker._encode_text_conditioning(
            "portrait", "", cfg=1.0, cache_scope="image", reuse_negative_at_cfg_one=True
        )

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertEqual(worker.clip.tokenize_calls, 1)
        self.assertEqual(worker.clip.encode_calls, 1)

    def test_does_not_reuse_conditioning_when_scope_changes(self):
        worker = object.__new__(ComfyWorker)
        worker.clip = _FakeClip()
        worker.nodes = type("Nodes", (), {"CLIPTextEncode": _FakeClipTextEncode})
        worker.clip_path = "clip.safetensors"
        worker.clip_type = "minimax"

        worker._encode_text_conditioning(
            "portrait", "", cfg=1.0, cache_scope="h3-t2v", reuse_negative_at_cfg_one=True
        )
        worker._encode_text_conditioning(
            "portrait", "", cfg=1.0, cache_scope="h3-ref2va", reuse_negative_at_cfg_one=True
        )

        self.assertEqual(worker.clip.tokenize_calls, 2)
        self.assertEqual(worker.clip.encode_calls, 2)


if __name__ == "__main__":
    unittest.main()
