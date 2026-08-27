import json
import unittest

from diffusion_workbench_api.settings import SettingsStore
from test_api_jobs import ApiTestCase


class SettingsTests(ApiTestCase):
    def test_returns_defaults_without_file(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["size_presets"],
            [[576, 576], [768, 768], [1024, 1024], [960, 1280]],
        )

    def test_update_persists_to_cache_dir(self):
        presets = [[576, 576], [512, 768]]
        response = self.client.put(
            "/api/v1/settings", json={"size_presets": presets}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["size_presets"], presets)

        settings_file = self.core.config.database.parent / "settings.json"
        self.assertTrue(settings_file.is_file())
        stored = json.loads(settings_file.read_text(encoding="utf-8"))
        self.assertEqual(stored["size_presets"], presets)

        # 新实例读取同一文件得到相同数据(持久化生效)
        reloaded = SettingsStore(settings_file)
        self.assertEqual(reloaded.get()["size_presets"], presets)

    def test_update_dedupes_and_rejects_invalid(self):
        response = self.client.put(
            "/api/v1/settings",
            json={"size_presets": [[576, 576], [576, 576]]},
        )
        self.assertEqual(response.json()["size_presets"], [[576, 576]])

        bad = self.client.put(
            "/api/v1/settings", json={"size_presets": [[577, 576]]}
        )
        self.assertEqual(bad.status_code, 422)

        unknown = self.client.put("/api/v1/settings", json={"nope": 1})
        self.assertEqual(unknown.status_code, 422)

    def test_task_sampling_params_are_not_global_settings(self):
        for key in ("negative_prompt", "cfg", "sampler", "scheduler", "steps"):
            response = self.client.put("/api/v1/settings", json={key: "x"})
            self.assertEqual(response.status_code, 422, f"{key} 不应成为全局设置")


class VideoSizePresetsTests(ApiTestCase):
    def test_returns_defaults_without_file(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["wan_video_size_presets"],
            [[704, 960], [960, 704], [1280, 720], [720, 1280]],
        )
        self.assertEqual(response.json()["minimax_video_size_presets"], [])

    def test_round_trip_persists(self):
        presets = [[704, 960], [1280, 720]]
        response = self.client.put(
            "/api/v1/settings", json={"wan_video_size_presets": presets}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["wan_video_size_presets"], presets)

        settings_file = self.core.config.database.parent / "settings.json"
        stored = json.loads(settings_file.read_text(encoding="utf-8"))
        self.assertEqual(stored["wan_video_size_presets"], presets)

        reloaded = SettingsStore(settings_file)
        self.assertEqual(reloaded.get()["wan_video_size_presets"], presets)

    def test_rejects_invalid_values(self):
        for payload in (
            {"wan_video_size_presets": "not-a-list"},
            {"wan_video_size_presets": [[577, 576]]},
            {"wan_video_size_presets": [[576]]},
            {"wan_video_size_presets": [[0, 576]]},
            {"wan_video_size_presets": [[576.0, 576]]},
        ):
            response = self.client.put("/api/v1/settings", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")

    def test_minimax_presets_require_multiple_of_32(self):
        presets = [[608, 352], [768, 1344]]
        response = self.client.put(
            "/api/v1/settings", json={"minimax_video_size_presets": presets}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["minimax_video_size_presets"], presets)

        # 16 的倍数但不是 32 的倍数应被拒绝
        bad = self.client.put(
            "/api/v1/settings", json={"minimax_video_size_presets": [[720, 352]]}
        )
        self.assertEqual(bad.status_code, 422)

    def test_legacy_video_size_presets_migrate_to_wan(self):
        settings_file = self.core.config.database.parent / "settings.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        legacy = [[704, 960], [960, 704]]
        settings_file.write_text(
            json.dumps({"video_size_presets": legacy}), encoding="utf-8"
        )

        store = SettingsStore(settings_file)
        self.assertEqual(store.get()["wan_video_size_presets"], legacy)

        # 新键已存在时旧键不覆盖
        settings_file.write_text(
            json.dumps(
                {
                    "video_size_presets": legacy,
                    "wan_video_size_presets": [[1280, 720]],
                }
            ),
            encoding="utf-8",
        )
        store = SettingsStore(settings_file)
        self.assertEqual(store.get()["wan_video_size_presets"], [[1280, 720]])


class Ref2vaLimitsTests(ApiTestCase):
    def test_returns_conservative_defaults(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["ref2va_limits"],
            {"max_images": 3, "max_videos": 1, "max_audios": 1},
        )

    def test_partial_update_merges_defaults(self):
        response = self.client.put(
            "/api/v1/settings", json={"ref2va_limits": {"max_images": 9}}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["ref2va_limits"],
            {"max_images": 9, "max_videos": 1, "max_audios": 1},
        )
        persisted = self.client.get("/api/v1/settings").json()
        self.assertEqual(persisted["ref2va_limits"]["max_images"], 9)

    def test_rejects_out_of_range_and_total_overflow(self):
        too_many = self.client.put(
            "/api/v1/settings", json={"ref2va_limits": {"max_images": 10}}
        )
        self.assertEqual(too_many.status_code, 422)
        overflow = self.client.put(
            "/api/v1/settings",
            json={"ref2va_limits": {"max_images": 9, "max_videos": 3, "max_audios": 3}},
        )
        # 9+3+3=15 超过总数 12 的硬上限
        self.assertEqual(overflow.status_code, 422)
        unknown = self.client.put(
            "/api/v1/settings", json={"ref2va_limits": {"max_text": 1}}
        )
        self.assertEqual(unknown.status_code, 422)


class LlmSettingsTests(ApiTestCase):
    def test_llm_defaults(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        llm = response.json()["llm"]
        self.assertEqual(llm["interface"], "ollama")
        self.assertEqual(llm["base_url"], "http://127.0.0.1:11434")
        self.assertEqual(llm["api_key"], "")
        self.assertEqual(llm["model"], "")
        self.assertEqual(llm["system_prompt"], "")
        self.assertIn("SDXL", llm["sd_system_prompt"])
        self.assertIn("逗号分隔短语", llm["sd_system_prompt"])
        self.assertIn("Positive:", llm["format_prompt"])
        self.assertIn("Negative:", llm["format_prompt"])
        self.assertIn("{language}", llm["language_prompt"])
        self.assertIs(llm["think"], False)
        self.assertEqual(llm["think_effort"], "")

    def test_llm_think_toggle(self):
        response = self.client.put("/api/v1/settings", json={"llm": {"think": True}})
        self.assertEqual(response.status_code, 200)
        self.assertIs(response.json()["llm"]["think"], True)
        response = self.client.put("/api/v1/settings", json={"llm": {"think": "yes"}})
        self.assertEqual(response.status_code, 422)

    def test_llm_partial_update_merges_defaults(self):
        response = self.client.put(
            "/api/v1/settings",
            json={"llm": {"interface": "openai", "model": "gpt-4o-mini"}},
        )
        self.assertEqual(response.status_code, 200)
        llm = response.json()["llm"]
        self.assertEqual(llm["interface"], "openai")
        self.assertEqual(llm["model"], "gpt-4o-mini")
        # 未提交的键回退默认值
        self.assertEqual(llm["base_url"], "http://127.0.0.1:11434")
        self.assertIn("{language}", llm["language_prompt"])

    def test_llm_prompt_styles_round_trip_independently(self):
        response = self.client.put(
            "/api/v1/settings",
            json={
                "llm": {
                    "system_prompt": "FLUX_CUSTOM",
                    "sd_system_prompt": "SD_CUSTOM",
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        llm = response.json()["llm"]
        self.assertEqual(llm["system_prompt"], "FLUX_CUSTOM")
        self.assertEqual(llm["sd_system_prompt"], "SD_CUSTOM")

        settings_file = self.core.config.database.parent / "settings.json"
        reloaded = SettingsStore(settings_file).get()["llm"]
        self.assertEqual(reloaded["system_prompt"], "FLUX_CUSTOM")
        self.assertEqual(reloaded["sd_system_prompt"], "SD_CUSTOM")

    def test_llm_rejects_invalid_interface(self):
        response = self.client.put(
            "/api/v1/settings", json={"llm": {"interface": "anthropic"}}
        )
        self.assertEqual(response.status_code, 422)

    def test_llm_rejects_wrong_types_and_unknown_keys(self):
        for payload in (
            {"llm": "not-a-dict"},
            {"llm": {"model": 123}},
            {"llm": {"nope": "x"}},
        ):
            response = self.client.put("/api/v1/settings", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")


class CaptionApiSettingsTests(ApiTestCase):
    def test_caption_api_defaults(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        caption_api = response.json()["caption_api"]
        self.assertEqual(caption_api["interface"], "ollama")
        self.assertEqual(caption_api["base_url"], "http://127.0.0.1:11434")
        self.assertEqual(caption_api["api_key"], "")
        self.assertEqual(caption_api["model"], "")
        self.assertIs(caption_api["think"], False)
        # 提示词默认值与本地反推的系统提示词一致(含双语输出要求)
        self.assertIn("image prompt engineer", caption_api["prompt"])
        self.assertIn("bilingual", caption_api["prompt"])

    def test_caption_api_partial_update_merges_defaults(self):
        response = self.client.put(
            "/api/v1/settings",
            json={"caption_api": {"interface": "openai", "model": "gpt-4o-mini"}},
        )
        self.assertEqual(response.status_code, 200)
        caption_api = response.json()["caption_api"]
        self.assertEqual(caption_api["interface"], "openai")
        self.assertEqual(caption_api["model"], "gpt-4o-mini")
        # 未提交的键回退默认值
        self.assertEqual(caption_api["base_url"], "http://127.0.0.1:11434")
        self.assertIn("image prompt engineer", caption_api["prompt"])

    def test_caption_api_think_toggle(self):
        response = self.client.put(
            "/api/v1/settings", json={"caption_api": {"think": True}}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIs(response.json()["caption_api"]["think"], True)
        response = self.client.put(
            "/api/v1/settings", json={"caption_api": {"think": "yes"}}
        )
        self.assertEqual(response.status_code, 422)

    def test_caption_api_rejects_invalid_interface(self):
        response = self.client.put(
            "/api/v1/settings", json={"caption_api": {"interface": "anthropic"}}
        )
        self.assertEqual(response.status_code, 422)

    def test_caption_api_rejects_wrong_types_and_unknown_keys(self):
        for payload in (
            {"caption_api": "not-a-dict"},
            {"caption_api": {"model": 123}},
            {"caption_api": {"nope": "x"}},
        ):
            response = self.client.put("/api/v1/settings", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")

    def test_caption_api_round_trip_persistence(self):
        response = self.client.put(
            "/api/v1/settings",
            json={"caption_api": {"model": "qwen3-vl:8b", "prompt": "CUSTOM"}},
        )
        self.assertEqual(response.status_code, 200)
        settings_file = self.core.config.database.parent / "settings.json"
        reloaded = SettingsStore(settings_file).get()["caption_api"]
        self.assertEqual(reloaded["model"], "qwen3-vl:8b")
        self.assertEqual(reloaded["prompt"], "CUSTOM")


class SamplingDefaultsTests(ApiTestCase):
    def test_returns_defaults_per_mode(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        defaults = response.json()["sampling_defaults"]
        for mode in ("zit", "krea2", "zib", "sdxl"):
            self.assertEqual(
                defaults[mode],
                {"steps": 8, "sampler": "euler", "scheduler": "simple", "cfg": 1},
            )

    def test_partial_update_merges_defaults(self):
        response = self.client.put(
            "/api/v1/settings",
            json={"sampling_defaults": {"zit": {"steps": 20, "sampler": "dpmpp_2m_sde"}}},
        )
        self.assertEqual(response.status_code, 200)
        defaults = response.json()["sampling_defaults"]
        self.assertEqual(defaults["zit"]["steps"], 20)
        self.assertEqual(defaults["zit"]["sampler"], "dpmpp_2m_sde")
        # 未提交字段/模式回退默认值
        self.assertEqual(defaults["zit"]["scheduler"], "simple")
        self.assertEqual(defaults["zit"]["cfg"], 1)
        self.assertEqual(defaults["krea2"]["steps"], 8)

    def test_rejects_invalid_values(self):
        for payload in (
            {"sampling_defaults": "not-a-dict"},
            {"sampling_defaults": {"nope": {"steps": 8}}},
            {"sampling_defaults": {"zit": "not-a-dict"}},
            {"sampling_defaults": {"zit": {"nope": 1}}},
            {"sampling_defaults": {"zit": {"steps": 0}}},
            {"sampling_defaults": {"zit": {"steps": 101}}},
            {"sampling_defaults": {"zit": {"steps": 8.5}}},
            {"sampling_defaults": {"zit": {"cfg": 0}}},
            {"sampling_defaults": {"zit": {"cfg": "1"}}},
            {"sampling_defaults": {"zit": {"sampler": "nope"}}},
            {"sampling_defaults": {"zit": {"scheduler": "nope"}}},
        ):
            response = self.client.put("/api/v1/settings", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")


class PromptPresetsTests(ApiTestCase):
    def test_defaults_to_empty_list(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt_presets"], [])

    def test_round_trip_persists(self):
        presets = [
            {"id": "a1", "title": "通用质量词", "kind": "positive", "text": "masterpiece, best quality"},
            {"id": "b2", "title": "通用负面", "kind": "negative", "text": "lowres, blurry"},
        ]
        response = self.client.put(
            "/api/v1/settings", json={"prompt_presets": presets}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt_presets"], presets)

        settings_file = self.core.config.database.parent / "settings.json"
        reloaded = SettingsStore(settings_file)
        self.assertEqual(reloaded.get()["prompt_presets"], presets)

    def test_rejects_invalid_items(self):
        for payload in (
            {"prompt_presets": "not-a-list"},
            {"prompt_presets": ["not-a-dict"]},
            {"prompt_presets": [{"id": "a", "title": "t", "kind": "positive", "text": "x", "nope": 1}]},
            {"prompt_presets": [{"id": "", "title": "t", "kind": "positive", "text": "x"}]},
            {"prompt_presets": [{"id": "a", "title": "  ", "kind": "positive", "text": "x"}]},
            {"prompt_presets": [{"id": "a", "title": "t", "kind": "neutral", "text": "x"}]},
            {"prompt_presets": [{"id": "a", "title": "t", "kind": "positive", "text": 1}]},
        ):
            response = self.client.put("/api/v1/settings", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")


if __name__ == "__main__":
    unittest.main()
