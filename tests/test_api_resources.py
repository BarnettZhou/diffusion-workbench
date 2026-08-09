import unittest

from diffusion_workbench_core.domain import Mode, ResourceKind

from test_api_jobs import ApiTestCase


class ResourcesTests(ApiTestCase):
    def test_modes_expose_server_loader_capabilities(self):
        response = self.client.get("/api/v1/modes")

        self.assertEqual(response.status_code, 200)
        modes = {item["mode"]: item for item in response.json()["modes"]}
        self.assertTrue(modes["zit"]["requires_vae"])
        self.assertEqual(modes["sdxl"]["model_loader"], "checkpoint")
        self.assertFalse(modes["sdxl"]["requires_vae"])

    def test_sdxl_lists_checkpoint_and_has_no_external_vae(self):
        models = self.client.get("/api/v1/resources/sdxl/diffusion")
        vaes = self.client.get("/api/v1/resources/sdxl/vae")

        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json()["resources"][0]["name"], "sdxl.safetensors")
        self.assertEqual(vaes.status_code, 200)
        self.assertEqual(vaes.json()["resources"], [])

    def test_list_resources_hides_server_paths(self):
        response = self.client.get("/api/v1/resources/zit/diffusion")

        self.assertEqual(response.status_code, 200)
        resources = response.json()["resources"]
        self.assertEqual(len(resources), 2)
        first = resources[0]
        self.assertEqual(first["index"], 1)
        self.assertEqual(first["name"], "zit.safetensors")
        self.assertIsNone(first["alias"])
        self.assertNotIn("path", first)
        self.assertNotIn(str(self.core.root), str(resources))

    def test_zib_resources_listed_without_paths(self):
        for kind in ("diffusion", "vae"):
            response = self.client.get(f"/api/v1/resources/zib/{kind}")
            self.assertEqual(response.status_code, 200)
            resources = response.json()["resources"]
            self.assertEqual(len(resources), 1)
            self.assertNotIn("path", resources[0])
            self.assertNotIn(str(self.core.root), str(resources))

    def test_invalid_mode_or_kind_returns_422(self):
        self.assertEqual(
            self.client.get("/api/v1/resources/sd15/diffusion").status_code, 422
        )
        self.assertEqual(
            self.client.get("/api/v1/resources/zit/lora").status_code, 422
        )

    def test_set_alias(self):
        response = self.client.put(
            "/api/v1/resources/zit/diffusion/1/alias", json={"alias": "portrait"}
        )
        self.assertEqual(response.status_code, 200)
        item = self.core.items[(Mode.ZIT, ResourceKind.DIFFUSION)][0]
        self.assertEqual(item.alias, "portrait")

    def test_set_alias_unknown_index_returns_404(self):
        response = self.client.put(
            "/api/v1/resources/zit/vae/99/alias", json={"alias": "x"}
        )
        self.assertEqual(response.status_code, 404)

    def test_conflicting_alias_returns_422(self):
        self.client.put(
            "/api/v1/resources/zit/diffusion/1/alias", json={"alias": "portrait"}
        )
        response = self.client.put(
            "/api/v1/resources/zit/diffusion/2/alias", json={"alias": "portrait"}
        )
        self.assertEqual(response.status_code, 422)


class StatusTests(ApiTestCase):
    def test_status_hides_pid_and_model_paths(self):
        response = self.client.get("/api/v1/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["worker"], "ready")
        self.assertEqual(body["gpu"], "1.0/16.0 GiB")
        self.assertNotIn("pid", body)
        self.assertNotIn("loaded_model", body)
        self.assertNotIn(str(self.core.root), str(body))


if __name__ == "__main__":
    unittest.main()
