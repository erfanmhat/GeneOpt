def pytest_addoption(parser):
    parser.addoption("--bench", action="store_true", help="also run the competitor benchmark smoke test")


def pytest_collection_modifyitems(config, items):
    # deselect (not skip) so `-m` filters in CI stay independent of this switch
    if config.getoption("--bench"):
        return
    bench = [item for item in items if "bench" in item.keywords]
    if bench:
        config.hook.pytest_deselected(items=bench)
        items[:] = [item for item in items if "bench" not in item.keywords]
