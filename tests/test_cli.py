from epidemicflow.cli import main


def test_single_run(tmp_path, capsys):
    main(["--grid-size", "12", "--init-infections", "2", "--seed", "1", "--output", str(tmp_path)])
    assert "Total infected" in capsys.readouterr().out
    assert len(list(tmp_path.glob("*.csv"))) == 1


def test_many_runs_with_plot(tmp_path, capsys):
    plot = tmp_path / "curves.png"
    main(["--grid-size", "12", "--init-infections", "2", "--runs", "4",
          "--output", str(tmp_path), "--plot", str(plot)])
    assert "4 runs" in capsys.readouterr().out
    assert plot.exists()
