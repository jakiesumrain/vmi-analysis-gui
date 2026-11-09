## Why

<!-- What problem does this solve? Link the issue if there is one. -->

## What changed

<!-- Bullet list of user-visible and internal changes. -->

## Test evidence

<!-- Exact commands run and their result. There is no CI, so this is the signal. -->

```text
uv run python -m unittest discover -s tests -v
# ...
```

## UI evidence

<!-- Screenshots/GIF for GUI changes; write "N/A" otherwise. -->

## Backward compatibility / risks

<!-- CLI flags, config keys, saved .nicegui keys, output paths, legacy data roots. -->

## Checklist

- [ ] Branch is off `main` and contains one topic
- [ ] `uv run python -m unittest discover -s tests -v` passes
- [ ] No datasets, analysis outputs, `.nicegui/`, `.cache/`, or unrelated edits staged
- [ ] No machine-specific paths added; scripts remain standalone-runnable
- [ ] GUI rules respected (single Run/Stop per panel, `gui/resources.py` in sync)
- [ ] Docs updated if behavior or workflow changed
- [ ] Owner review requested; PR not merged by the contributor
