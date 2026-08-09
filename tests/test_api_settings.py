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
