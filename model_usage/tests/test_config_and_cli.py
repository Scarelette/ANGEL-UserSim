import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from model_usage.angel.cli import build_config, build_parser, main, profile_kwargs
from model_usage.angel.config import ActorConfig, ObserverConfig, RunnerConfig

ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / "examples" / "profiles.jsonl"


class TestConfigResolution(unittest.TestCase):
    def test_defaults_point_at_example_profiles(self):
        config = RunnerConfig()
        self.assertTrue(str(config.jsonl_path).endswith("examples/profiles.jsonl"))

    def test_env_var_overrides_default(self):
        with mock.patch.dict(os.environ, {"ANGEL_ACTOR_MODEL": "/tmp/some-model"}):
            self.assertEqual(ActorConfig().model_path, "/tmp/some-model")

    def test_generation_settings_are_the_demo_defaults(self):
        actor, observer = ActorConfig(model_path="x"), ObserverConfig(model_path="x")
        self.assertEqual((actor.max_new_tokens, actor.temperature, actor.top_p), (90, 0.8, 0.9))
        self.assertEqual((observer.max_new_tokens, observer.temperature, observer.top_p), (3072, 0.7, 0.9))

    def test_backend_env_var(self):
        with mock.patch.dict(os.environ, {"ANGEL_BACKEND": "hf"}):
            self.assertEqual(RunnerConfig().backend, "hf")


class TestBackendSelection(unittest.TestCase):
    """`auto` must prefer vLLM and fall back to transformers."""

    def test_default_is_auto(self):
        self.assertEqual(RunnerConfig().backend, "auto")

    def test_explicit_kinds_construct(self):
        from model_usage.angel.backends import HFBackend, StubBackend, VLLMBackend, build_backend

        self.assertIsInstance(build_backend("stub", "/x"), StubBackend)
        self.assertIsInstance(build_backend("hf", "/x"), HFBackend)
        self.assertIsInstance(build_backend("vllm", "/x"), VLLMBackend)

    def test_unknown_kind_rejected(self):
        from model_usage.angel.backends import build_backend

        with self.assertRaises(ValueError):
            build_backend("tensorrt", "/x")

    def test_auto_picks_vllm_when_available(self):
        from model_usage.angel import backends

        with mock.patch.object(backends, "vllm_available", return_value=True):
            self.assertEqual(backends.build_backend("auto", "/x").name, "vllm")

    def test_auto_falls_back_to_hf(self):
        from model_usage.angel import backends

        with mock.patch.object(backends, "vllm_available", return_value=False):
            self.assertEqual(backends.build_backend("auto", "/x").name, "hf")

    def test_resolved_backend_reports_the_concrete_engine(self):
        config = RunnerConfig(backend="stub")
        self.assertEqual(config.resolved_backend(), "stub")
        self.assertIn(RunnerConfig(backend="auto").resolved_backend(), {"vllm", "hf"})

    def test_vllm_engine_defaults_match_patient_demo(self):
        # vLLM defaults: 40% GPU memory, 8192-token context.
        config = RunnerConfig()
        self.assertEqual(config.vllm_gpu_memory_utilization, 0.4)
        self.assertEqual(config.vllm_max_model_len, 8192)

    def test_vllm_backend_is_lazy(self):
        from model_usage.angel.backends import VLLMBackend

        self.assertFalse(VLLMBackend("/nonexistent").loaded)


class TestPreflight(unittest.TestCase):
    def base(self, **kwargs):
        return RunnerConfig(
            observer=ObserverConfig(model_path="/nonexistent/observer"),
            actor=ActorConfig(model_path="/nonexistent/actor"),
            jsonl_path=JSONL,
            **kwargs,
        )

    def test_missing_actor_reported(self):
        problems = self.base().missing_paths(need_observer=False, need_actor=True)
        self.assertTrue(any("actor model not found" in p for p in problems))

    def test_missing_observer_reported_only_when_needed(self):
        config = self.base()
        self.assertFalse(
            any("observer" in p for p in config.missing_paths(need_observer=False, need_actor=False))
        )
        self.assertTrue(
            any("observer" in p for p in config.missing_paths(need_observer=True, need_actor=False))
        )

    def test_stub_backend_needs_no_weights(self):
        config = self.base(backend="stub")
        self.assertEqual(config.missing_paths(need_observer=True, need_actor=True), [])
        config.preflight(need_observer=True, need_actor=True)  # must not raise

    def test_missing_jsonl_reported(self):
        config = RunnerConfig(jsonl_path=Path("/nonexistent/patients.jsonl"), backend="stub")
        problems = config.missing_paths(need_observer=False, need_actor=False)
        self.assertTrue(any("profiles JSONL not found" in p for p in problems))

    def test_preflight_raises_readable_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            self.base().preflight(need_actor=True)
        self.assertIn("ANGEL_ACTOR_MODEL", str(ctx.exception))

    def test_describe_is_serializable(self):
        json.dumps(self.base().describe())


class TestCliArgWiring(unittest.TestCase):
    def parse(self, argv):
        return build_parser().parse_args(argv)

    def test_flags_override_config(self):
        args = self.parse(
            ["chat", "--actor-model", "/x/actor", "--observer-model", "/x/obs",
             "--backend", "stub", "--keep-both"]
        )
        config = build_config(args)
        self.assertEqual(config.actor.model_path, "/x/actor")
        self.assertEqual(config.observer.model_path, "/x/obs")
        self.assertEqual(config.backend, "stub")
        self.assertTrue(config.keep_both_resident)

    def test_profile_sources_are_mutually_exclusive(self):
        # argparse prints its usage to stderr before exiting; keep test output clean.
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse(["chat", "--profile-id", "0", "--short-profile", "x"])

    def test_profile_id_maps_to_send_kwargs(self):
        self.assertEqual(profile_kwargs(self.parse(["chat", "--profile-id", "3"])),
                         {"profile_id": "3", "expand": True})

    def test_default_is_first_builtin_profile(self):
        self.assertEqual(profile_kwargs(self.parse(["chat"])), {"profile_id": "0", "expand": True})

    def test_expand_flag_is_forwarded(self):
        self.assertFalse(profile_kwargs(self.parse(["chat", "--profile-id", "3", "--no-expand"]))["expand"])

    def test_short_profile_needs_no_expand_flag(self):
        # --short-profile always runs stage 1, so it carries no `expand` key.
        kwargs = profile_kwargs(self.parse(["chat", "--short-profile", "Mara, 34."]))
        self.assertNotIn("expand", kwargs)

    def test_short_profile_maps_to_send_kwargs(self):
        kwargs = profile_kwargs(self.parse(["chat", "--short-profile", "Mara, 34."]))
        self.assertEqual(kwargs, {"short_profile": "Mara, 34."})

    def test_profile_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps({"identity": {"name": "Z"}}), encoding="utf-8")
            kwargs = profile_kwargs(self.parse(["chat", "--profile-file", str(path)]))
            self.assertEqual(kwargs["profile"]["identity"]["name"], "Z")

    def test_missing_profile_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            profile_kwargs(self.parse(["chat", "--profile-file", "/nonexistent/p.json"]))

    def test_short_profile_file_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.txt"
            path.write_text("Mara, 34, exhausted.", encoding="utf-8")
            kwargs = profile_kwargs(self.parse(["chat", "--short-profile-file", str(path)]))
            self.assertIn("Mara", kwargs["short_profile"])


class TestCliEndToEnd(unittest.TestCase):
    """Drives the real CLI against the stub backend — no GPU, no weights."""

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_list_profiles(self):
        code, out, _ = self.run_cli(["--list"])
        self.assertEqual(code, 0)
        self.assertIn("Profiles in ", out)

    def test_status(self):
        code, out, _ = self.run_cli(["--status"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["ok"])

    def test_say_one_message(self):
        code, out, err = self.run_cli(
            ["say", "--backend", "stub", "--jsonl", str(JSONL), "--profile-id", "0", "Hello there."]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("[patient]", out)

    def test_say_through_the_observer(self):
        code, out, err = self.run_cli(
            ["say", "--backend", "stub", "--jsonl", str(JSONL),
             "--short-profile", "Mara, 34, running on empty since spring.", "Hello."]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("observer+actor", out)

    def test_expand_writes_a_rich_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "rich.json"
            raw_path = Path(tmp) / "raw.json"
            code, _, err = self.run_cli(
                ["expand", "--backend", "stub", "--jsonl", str(JSONL),
                 "--short-profile", "Mara, 34, exhausted.",
                 "--out", str(out_path), "--raw-out", str(raw_path)]
            )
            self.assertEqual(code, 0, err)
            rich = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("identity", rich)
            self.assertIn("symptom_details", json.loads(raw_path.read_text(encoding="utf-8")))

    def test_expand_without_short_profile_fails_cleanly(self):
        code, _, err = self.run_cli(
            ["expand", "--backend", "stub", "--jsonl", str(JSONL), "--profile-id", "0"]
        )
        self.assertEqual(code, 2)
        self.assertIn("--short-profile", err)

    def test_missing_actor_weights_fails_with_guidance(self):
        code, _, err = self.run_cli(
            ["say", "--backend", "hf", "--jsonl", str(JSONL),
             "--actor-model", "/nonexistent/actor", "--profile-id", "0", "--no-expand", "Hello."]
        )
        self.assertEqual(code, 2)
        self.assertIn("ANGEL_ACTOR_MODEL", err)

    def test_bad_profile_id_fails_cleanly(self):
        code, _, err = self.run_cli(
            ["say", "--backend", "stub", "--jsonl", str(JSONL), "--profile-id", "9999", "Hello."]
        )
        self.assertEqual(code, 1)
        self.assertIn("out of range", err)

    def test_no_args_prints_help(self):
        code, out, _ = self.run_cli([])
        self.assertEqual(code, 0)
        self.assertIn("usage:", out)


if __name__ == "__main__":
    unittest.main()
