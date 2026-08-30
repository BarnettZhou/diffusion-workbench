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


class AspectRatioPresetsTests(ApiTestCase):
    def test_returns_defaults_without_file(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["aspect_ratio_presets"],
            [[1, 1], [3, 4], [4, 3], [5, 4], [4, 5], [16, 9], [9, 16]],
        )

    def test_round_trip_persists_and_dedupes(self):
        presets = [[3, 4], [16, 9], [3, 4]]
        response = self.client.put(
            "/api/v1/settings", json={"aspect_ratio_presets": presets}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["aspect_ratio_presets"], [[3, 4], [16, 9]])

        settings_file = self.core.config.database.parent / "settings.json"
        reloaded = SettingsStore(settings_file)
        self.assertEqual(reloaded.get()["aspect_ratio_presets"], [[3, 4], [16, 9]])

    def test_rejects_invalid_values(self):
        for payload in (
            {"aspect_ratio_presets": "not-a-list"},
            {"aspect_ratio_presets": [[3]]},
            {"aspect_ratio_presets": [[0, 4]]},
            {"aspect_ratio_presets": [[3.0, 4]]},
            {"aspect_ratio_presets": [[True, 4]]},
        ):
            response = self.client.put("/api/v1/settings", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")


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
    def _endpoint(self, **overrides):
        endpoint = {
            "id": "a1b2c3d4",
            "name": "本地 Ollama",
            "interface": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "api_key": "",
            "models": ["qwen3"],
        }
        endpoint.update(overrides)
        return endpoint

    def _selected(self, model="qwen3"):
        return {"endpoint_id": "a1b2c3d4", "model": model}

    def test_llm_defaults(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        llm = response.json()["llm"]
        self.assertEqual(llm["endpoints"], [])
        self.assertEqual(llm["selected"], {"endpoint_id": "", "model": ""})
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
            json={
                "llm": {
                    "endpoints": [self._endpoint(interface="openai")],
                    "selected": self._selected(model="gpt-4o-mini"),
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        llm = response.json()["llm"]
        self.assertEqual(llm["endpoints"][0]["interface"], "openai")
        self.assertEqual(llm["selected"]["model"], "gpt-4o-mini")
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
            "/api/v1/settings",
            json={"llm": {"endpoints": [self._endpoint(interface="anthropic")]}},
        )
        self.assertEqual(response.status_code, 422)

    def test_llm_rejects_wrong_types_and_unknown_keys(self):
        for payload in (
            {"llm": "not-a-dict"},
            {"llm": {"endpoints": "not-a-list"}},
            {"llm": {"endpoints": [self._endpoint(models=[1])]}},
            {"llm": {"nope": "x"}},
        ):
            response = self.client.put("/api/v1/settings", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")

    def test_llm_rejects_unknown_endpoint_keys_and_blank_base_url(self):
        endpoint = self._endpoint(extra="x")
        bad = self.client.put("/api/v1/settings", json={"llm": {"endpoints": [endpoint]}})
        self.assertEqual(bad.status_code, 422)
        bad = self.client.put(
            "/api/v1/settings",
            json={"llm": {"endpoints": [self._endpoint(base_url="  ")]}},
        )
        self.assertEqual(bad.status_code, 422)

    def test_legacy_flat_llm_migrates_on_update_and_file_load(self):
        response = self.client.put(
            "/api/v1/settings",
            json={
                "llm": {
                    "interface": "openai",
                    "base_url": "https://api.example/v1",
                    "api_key": "secret",
                    "model": "gpt-4o-mini",
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        llm = response.json()["llm"]
        self.assertEqual(llm["endpoints"][0]["name"], "")
        self.assertEqual(llm["endpoints"][0]["models"], ["gpt-4o-mini"])
        self.assertEqual(llm["selected"], {"endpoint_id": llm["endpoints"][0]["id"], "model": "gpt-4o-mini"})
        self.assertNotIn("model", llm)

        settings_file = self.core.config.database.parent / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "llm": {
                        "interface": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        "api_key": "",
                        "model": "qwen3",
                    }
                }
            ),
            encoding="utf-8",
        )
        reloaded = SettingsStore(settings_file).get()["llm"]
        self.assertEqual(reloaded["endpoints"][0]["models"], ["qwen3"])
        self.assertNotIn("model", reloaded)

    def test_selected_endpoint_must_exist(self):
        response = self.client.put(
            "/api/v1/settings",
            json={
                "llm": {
                    "endpoints": [self._endpoint()],
                    "selected": {"endpoint_id": "missing", "model": "qwen3"},
                }
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_endpoint_with_local_temp_id_gets_generated_id(self):
        # 前端本地新建的端点带 local-* 临时 id,保存时服务端应重新生成而非报错
        response = self.client.put(
            "/api/v1/settings",
            json={"llm": {"endpoints": [self._endpoint(id="local-1730000000000")]}},
        )
        self.assertEqual(response.status_code, 200)
        endpoint = response.json()["llm"]["endpoints"][0]
        self.assertEqual(len(endpoint["id"]), 8)
        int(endpoint["id"], 16)

    def test_selected_only_update_preserves_endpoints(self):
        # 前端切换默认模型只提交 selected;endpoints 等其余键必须保留
        self.client.put(
            "/api/v1/settings", json={"llm": {"endpoints": [self._endpoint()]}}
        )
        response = self.client.put(
            "/api/v1/settings", json={"llm": {"selected": self._selected()}}
        )
        self.assertEqual(response.status_code, 200)
        llm = response.json()["llm"]
        self.assertEqual(llm["selected"], self._selected())
        self.assertEqual(len(llm["endpoints"]), 1)
        self.assertEqual(llm["endpoints"][0]["models"], ["qwen3"])


class CaptionApiSettingsTests(ApiTestCase):
    def _endpoint(self, **overrides):
        endpoint = {
            "id": "c1d2e3f4",
            "name": "视觉模型",
            "interface": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "api_key": "",
            "models": ["qwen3-vl:8b"],
        }
        endpoint.update(overrides)
        return endpoint

    def _selected(self, model="qwen3-vl:8b"):
        return {"endpoint_id": "c1d2e3f4", "model": model}

    def test_caption_api_defaults(self):
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        caption_api = response.json()["caption_api"]
        self.assertEqual(caption_api["endpoints"], [])
        self.assertEqual(caption_api["selected"], {"endpoint_id": "", "model": ""})
        self.assertIs(caption_api["think"], False)
        self.assertIn("image prompt engineer", caption_api["prompt"])
        self.assertIn("bilingual", caption_api["prompt"])

    def test_caption_api_partial_update_merges_defaults(self):
        response = self.client.put(
            "/api/v1/settings",
            json={
                "caption_api": {
                    "endpoints": [self._endpoint(interface="openai")],
                    "selected": self._selected(model="gpt-4o-mini"),
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        caption_api = response.json()["caption_api"]
        self.assertEqual(caption_api["endpoints"][0]["interface"], "openai")
        self.assertEqual(caption_api["selected"]["model"], "gpt-4o-mini")
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
            "/api/v1/settings",
            json={"caption_api": {"endpoints": [self._endpoint(interface="anthropic")]}},
        )
        self.assertEqual(response.status_code, 422)

    def test_caption_api_rejects_wrong_types_and_unknown_keys(self):
        for payload in (
            {"caption_api": "not-a-dict"},
            {"caption_api": {"endpoints": "not-a-list"}},
            {"caption_api": {"endpoints": [self._endpoint(models=[1])]}},
            {"caption_api": {"nope": "x"}},
        ):
            response = self.client.put("/api/v1/settings", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")

    def test_legacy_flat_caption_api_migrates_on_update_and_file_load(self):
        response = self.client.put(
            "/api/v1/settings",
            json={
                "caption_api": {
                    "interface": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "api_key": "",
                    "model": "qwen3-vl:8b",
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        caption_api = response.json()["caption_api"]
        self.assertEqual(caption_api["endpoints"][0]["models"], ["qwen3-vl:8b"])
        self.assertEqual(caption_api["selected"]["model"], "qwen3-vl:8b")
        self.assertNotIn("model", caption_api)

        settings_file = self.core.config.database.parent / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "caption_api": {
                        "interface": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        "api_key": "",
                        "model": "qwen3-vl:8b",
                    }
                }
            ),
            encoding="utf-8",
        )
        reloaded = SettingsStore(settings_file).get()["caption_api"]
        self.assertEqual(reloaded["endpoints"][0]["models"], ["qwen3-vl:8b"])
        self.assertNotIn("model", reloaded)

    def test_caption_api_round_trip_persistence(self):
        response = self.client.put(
            "/api/v1/settings",
            json={
                "caption_api": {
                    "endpoints": [self._endpoint()],
                    "selected": self._selected(),
                    "prompt": "CUSTOM",
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        settings_file = self.core.config.database.parent / "settings.json"
        reloaded = SettingsStore(settings_file).get()["caption_api"]
        self.assertEqual(reloaded["selected"]["model"], "qwen3-vl:8b")
        self.assertEqual(reloaded["prompt"], "CUSTOM")

    def test_selected_only_update_preserves_endpoints(self):
        # 前端切换默认模型只提交 selected;endpoints 等其余键必须保留
        self.client.put(
            "/api/v1/settings",
            json={"caption_api": {"endpoints": [self._endpoint()]}},
        )
        response = self.client.put(
            "/api/v1/settings",
            json={"caption_api": {"selected": self._selected()}},
        )
        self.assertEqual(response.status_code, 200)
        caption_api = response.json()["caption_api"]
        self.assertEqual(caption_api["selected"], self._selected())
        self.assertEqual(len(caption_api["endpoints"]), 1)
        self.assertEqual(caption_api["endpoints"][0]["models"], ["qwen3-vl:8b"])


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
