import json

import pyarrow.parquet as pq

from scripts.build_verl_driver import driver_rows, write_driver


def test_driver_threads_task_and_seed():
    rows = driver_rows([{"task_id": "task-a"}, {"task_id": "task-b"}], seed=7)
    assert rows[0]["env_kwargs"] == {"task_id": "task-a", "seed": 7}
    assert rows[1]["env_kwargs"] == {"task_id": "task-b", "seed": 8}


def test_driver_writes_verl_parquet(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"task_id": "task-a"}) + "\n", encoding="utf-8")
    output = tmp_path / "train.parquet"
    write_driver(manifest, output, seed=11)
    row = pq.read_table(output).to_pylist()[0]
    assert row["prompt"] == [{"role": "user", "content": ""}]
    assert row["env_kwargs"] == {"task_id": "task-a", "seed": 11}
