"""Pure helper library shared by the lib-3 behaviour engines.

Everything under this package is side-effect free maths or static data: no
module here imports an engine, and only `gating` / `featcache` / `replay`
touch a MetricStore (passed in as an argument, duck-typed). Engines depend on
these helpers by name so that ~36 engines can be written in parallel against
one frozen contract (docs/lib3/helpers_api.md).

Import submodules explicitly (``from app.engines.behavior.lib import bayes``);
this package deliberately imports nothing eagerly so that a broken stub never
breaks an unrelated engine.
"""
