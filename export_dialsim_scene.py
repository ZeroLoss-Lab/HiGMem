import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one DialSim scene script to a readable .txt file.")
    default_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HiGMem_Other", "dialsim_v1.1.zip"))
    parser.add_argument(
        "--dialsim_source",
        type=str,
        default=default_source,
        help="Path to DialSim zip (Deflate64) or extracted directory containing *.pickle files.",
    )
    parser.add_argument("--show", type=str, required=True, help="friends | bigbang | theoffice")
    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Exact episode key in the pickle (e.g., 'S01E01 Monica Gets A Roommate.txt').",
    )
    parser.add_argument("--scene_id", type=int, required=True, help="Scene id (integer key inside the episode).")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output txt path. Default: H-Mem/dialsim_scene_examples/<show>_<episode>_scene<id>.txt",
    )
    args = parser.parse_args()

    # Local import to avoid slowing down other entrypoints.
    from dialsim_dataset import load_pickle_from_source, parse_script_to_turns

    third_party_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "third_party", "zipfile_deflate64"))
    show_pickle = f"{args.show}_dialsim.pickle"
    data = load_pickle_from_source(args.dialsim_source, show_pickle, third_party_dir=third_party_dir)

    if args.episode not in data:
        available = list(data.keys())[:10]
        raise SystemExit(f"Episode not found: {args.episode}. First 10 available: {available}")
    scenes = data[args.episode]
    if args.scene_id not in scenes:
        available = sorted(list(scenes.keys()))
        raise SystemExit(f"Scene id not found: {args.scene_id}. Available scene ids: {available[:50]}")

    item = scenes[args.scene_id]
    date = str(item.get("date", "") or "")
    script = str(item.get("script", "") or "")
    turns = parse_script_to_turns(script)

    if args.out is None:
        safe_ep = (
            args.episode.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dialsim_scene_examples"))
        os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, f"{args.show}_{safe_ep}_scene{args.scene_id}.txt")

    header = [
        f"show: {args.show}",
        f"episode: {args.episode}",
        f"scene_id: {args.scene_id}",
        f"date: {date}",
        f"num_turns: {len(turns)}",
        "",
        "# Parsed turns (speaker: text)",
    ]

    lines = []
    for i, (speaker, text) in enumerate(turns):
        lines.append(f"[{i:04d}] {speaker}: {text}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines))

    print(args.out)


if __name__ == "__main__":
    main()

