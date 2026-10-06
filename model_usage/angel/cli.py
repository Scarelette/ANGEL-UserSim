"""Command line entry point.

    python -m model_usage.angel chat --short-profile-file model_usage/examples/example_short_profile.txt
    python -m model_usage.angel say --short-profile "Mara, 34, exhausted since spring." "Hi, how are you?"
    python -m model_usage.angel expand --short-profile-file notes.txt --out profile.json
    python -m model_usage.angel --list

In-chat commands: /reset /history /profiles /status /quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .backends import NoGPUError
from .config import ActorConfig, ObserverConfig, RunnerConfig
from .pipeline import AngelModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m model_usage.angel",
        description="Run the two-stage Angel simulated-patient model directly (no server).",
    )
    parser.add_argument("--list", action="store_true", help="List the example profiles and exit.")
    parser.add_argument("--status", action="store_true", help="Print resolved configuration and exit.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--user", default="local", help="Conversation owner (keeps sessions separate).")
    common.add_argument("--session-id", default="default")
    common.add_argument("--backend", choices=["auto", "vllm", "hf", "stub"], default=None,
                        help="'auto' (default) prefers vLLM, then transformers; 'vllm' or 'hf' force "
                             "one engine; 'stub' is a no-GPU fake for trying the plumbing.")
    common.add_argument("--observer-model", default=None, help="Path to the stage-1 Observer checkpoint.")
    common.add_argument("--actor-model", default=None, help="Path to the stage-2 Actor checkpoint.")
    common.add_argument("--jsonl", default=None, help="Path to the profiles JSONL.")
    common.add_argument("--keep-both", action="store_true",
                        help="Keep the Observer resident after expansion (needs ~32 GB VRAM).")
    common.add_argument("--seed", type=int, default=None, help="Seed the reply-length RNG.")

    profile_args = argparse.ArgumentParser(add_help=False)
    group = profile_args.add_mutually_exclusive_group()
    group.add_argument("--profile-id", default=None, help="Profile from --jsonl: numeric index or canonical id.")
    group.add_argument("--profile-file", default=None, help="Rich-schema profile JSON file.")
    group.add_argument("--short-profile", default=None, help="Short free-text description (runs the Observer).")
    group.add_argument("--short-profile-file", default=None, help="File holding a short description.")
    profile_args.add_argument("--source-title", default="", help="Provenance label recorded on the profile.")
    profile_args.add_argument("--no-expand", dest="expand", action="store_false",
                              help="Skip the Observer for a --profile-file that is already an Observer "
                                   "expansion (the --out of `expand`). By default every profile is "
                                   "expanded first, so the Actor role-plays the long profile.")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", parents=[common, profile_args], help="Interactive conversation.")

    once = sub.add_parser("say", parents=[common, profile_args], help="Send one message and exit.")
    once.add_argument("message", help="The therapist message to send.")


    expand = sub.add_parser("expand", parents=[common, profile_args],
                            help="Run stage 1 only and write the rich profile.")
    expand.add_argument("--out", default=None, help="Write the rich profile JSON here (default: stdout).")
    expand.add_argument("--raw-out", default=None, help="Also write the Observer's untouched JSON here.")

    return parser


def build_config(args: argparse.Namespace) -> RunnerConfig:
    """Assemble config from the command-line flags (defaults in config.py)."""
    observer = ObserverConfig()
    actor = ActorConfig()

    if getattr(args, "observer_model", None):
        observer.model_path = args.observer_model
    if getattr(args, "actor_model", None):
        actor.model_path = args.actor_model

    config = RunnerConfig(observer=observer, actor=actor)
    if getattr(args, "jsonl", None):
        config.jsonl_path = Path(args.jsonl)
    if getattr(args, "backend", None):
        config.backend = args.backend
    if getattr(args, "keep_both", False):
        config.keep_both_resident = True
    if getattr(args, "seed", None) is not None:
        config.seed = args.seed
    return config


def profile_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Turn the profile flags into `AngelModel.send` keyword arguments."""
    if getattr(args, "profile_file", None):
        path = Path(args.profile_file)
        if not path.exists():
            raise FileNotFoundError(f"profile file not found: {path}")
        return {"profile": json.loads(path.read_text(encoding="utf-8")),
                "expand": bool(getattr(args, "expand", True))}

    if getattr(args, "short_profile_file", None):
        path = Path(args.short_profile_file)
        if not path.exists():
            raise FileNotFoundError(f"short profile file not found: {path}")
        return {"short_profile": path.read_text(encoding="utf-8")}

    if getattr(args, "short_profile", None):
        return {"short_profile": args.short_profile}

    if getattr(args, "profile_id", None):
        return {"profile_id": args.profile_id, "expand": bool(getattr(args, "expand", True))}

    # Nothing specified: use the first profile in --jsonl.
    return {"profile_id": "0", "expand": bool(getattr(args, "expand", True))}


def print_profiles(model: AngelModel) -> None:
    print(f"Profiles in {model.config.jsonl_path}:")
    for item in model.list_profiles():
        print(f"  {item['id']:>3}  {item['label']}")


def print_reply(result: Dict[str, Any], *, details: bool = True) -> None:
    print(f"\n[patient] {result['reply']}")
    if details:
        session = result["session"]
        stages = "observer+actor" if result["model"]["observer_used"] else "actor"
        print(f"    ({stages}, turn {session['num_turns']}, profile {session['profile'].get('name')})")


def cmd_expand(model: AngelModel, args: argparse.Namespace) -> int:
    kwargs = profile_kwargs(args)
    short = kwargs.get("short_profile")
    if not short:
        print("[error] expand needs --short-profile or --short-profile-file", file=sys.stderr)
        return 2

    result = model.expand_profile(short, source_title=args.source_title)
    payload = json.dumps(result.rich_profile, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(f"[ok] rich profile written to {args.out} (attempts: {result.attempts})")
    else:
        print(payload)

    if args.raw_out:
        Path(args.raw_out).write_text(
            json.dumps(result.observer_profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[ok] observer JSON written to {args.raw_out}")
    return 0


def cmd_say(model: AngelModel, args: argparse.Namespace) -> int:
    result = model.send(args.user, args.message, session_id=args.session_id, **profile_kwargs(args))
    print_reply(result)
    return 0


def cmd_chat(model: AngelModel, args: argparse.Namespace) -> int:
    print("\nInteractive chat. Commands: /reset /history /profiles /status /quit\n")
    first = True
    while True:
        try:
            message = input("[you] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message:
            continue
        if message == "/quit":
            break
        if message == "/profiles":
            print_profiles(model)
            continue
        if message == "/status":
            print(json.dumps(model.status(), indent=2))
            continue
        if message == "/history":
            try:
                for turn in model.history(args.user, session_id=args.session_id)["history"]:
                    speaker = "you" if turn["role"] == "user" else "patient"
                    print(f"  {speaker}: {turn['content']}")
            except KeyError:
                print("[error] no conversation yet.", file=sys.stderr)
            continue
        if message == "/reset":
            try:
                model.reset(args.user, session_id=args.session_id)
                print("[ok] conversation reset.")
            except KeyError:
                print("[error] no conversation yet.", file=sys.stderr)
            continue

        if first:
            print("   (loading the model — the first turn can take a while...)")
        try:
            result = model.send(
                args.user, message, session_id=args.session_id, **(profile_kwargs(args) if first else {})
            )
        except (ModuleNotFoundError, FileNotFoundError, NoGPUError) as exc:
            # Setup problems (missing package / model): retrying won't help.
            print(f"[error] {describe_error(exc)}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"[error] {describe_error(exc)}", file=sys.stderr)
            continue
        # Just the reply mid-conversation; /status has the details.
        print_reply(result, details=False)
        first = False

    model.end(args.user, session_id=args.session_id)
    print("[done] conversation ended.")
    return 0


_INSTALL_HINT = "install the requirements first:  pip install -r model_usage/requirements-usage.txt"


def describe_error(exc: BaseException) -> str:
    """Error text for the user, with a fix when we know one."""
    if isinstance(exc, ModuleNotFoundError):
        return f"missing Python package '{exc.name}'; {_INSTALL_HINT}"
    return str(exc)


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Bare --list / --status work without a subcommand.
    if args.command is None:
        if args.list or args.status:
            model = AngelModel(RunnerConfig())
            if args.list:
                print_profiles(model)
            if args.status:
                print(json.dumps(model.status(), indent=2))
            return 0
        parser.print_help()
        return 0

    config = build_config(args)
    model = AngelModel(config)

    try:
        if args.command == "expand":
            return cmd_expand(model, args)
        if args.command == "say":
            return cmd_say(model, args)
        if args.command == "chat":
            return cmd_chat(model, args)
    except (ModuleNotFoundError, FileNotFoundError, NoGPUError) as exc:
        print(f"[error] {describe_error(exc)}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        model.close()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
