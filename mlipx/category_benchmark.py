from typing import Dict, List, Any, Optional
import os
import zntrack
import pandas as pd

class CategoryBenchmark(zntrack.Node):
    """Generic node to combine a category of benchmarks."""
    benchmark_lists: Dict[str, List[Any]] = zntrack.deps()

    def run(self):
        pass

    @staticmethod
    def benchmark_precompute(
        benchmark_data_dict: Dict[str, List[Any] | Dict[str, Any]],
        benchmark_classes: Dict[str, Any],
        cache_dir: str,
        normalise_to_model: Optional[str] = None,
    ):
        from mlipx.dash_utils import process_data, compute_combined_score_table
        os.makedirs(cache_dir, exist_ok=True)

        maes = []
        for key, data in benchmark_data_dict.items():
            cls = benchmark_classes[key]
            data_dict = process_data(
                data,
                key_extractor=lambda node: node.name.split(f"_{key}")[0],
                value_extractor=lambda node: node,
            )
            cls.benchmark_precompute(node_dict=data_dict, normalise_to_model=normalise_to_model)
            mae_path = os.path.join(cache_dir, f"{key}_cache/mae_df.pkl")
            maes.append(pd.read_pickle(mae_path))

        combined_score = compute_combined_score_table(maes)
        combined_score.to_pickle(os.path.join(cache_dir, "benchmark_score.pkl"))

    @staticmethod
    def launch_dashboard(
        cache_dir: str,
        benchmark_keys: List[str],
        benchmark_classes: Dict[str, Any],
        benchmark_titles: Dict[str, str],
        normalise_to_model: Optional[str] = None,
        ui=None,
        full_benchmark: bool = False,
    ):
        import pandas as pd
        import dash
        import pickle
        from mlipx.dash_utils import run_app, combine_apps

        benchmark_score_df = pd.read_pickle(os.path.join(cache_dir, "benchmark_score.pkl"))
        layouts = []

        for key in benchmark_keys:
            mae_df = pd.read_pickle(os.path.join(cache_dir, f"{key}_cache/mae_df.pkl"))
            layouts.append(benchmark_classes[key].build_layout(mae_df))

        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title=" + ".join(benchmark_titles.values()),
            apps_or_layouts_list=layouts,
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
            id=f"{'_'.join(benchmark_keys)}-score-table",
            static_coloured_table=True,
        )

        if full_benchmark:
            with open(os.path.join(cache_dir, "callback_data.pkl"), "rb") as f:
                callback_fn = pickle.load(f)
            return layout, callback_fn

        app.layout = layout
        with open(os.path.join(cache_dir, "callback_data.pkl"), "rb") as f:
            callback_fn = pickle.load(f)
        callback_fn(app)
        return run_app(app, ui=ui)
