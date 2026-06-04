# tests

Holds the **regression anchor** for the `refactor/lmfi-migration` work: the
current-code numbers each migrated experiment must reproduce (see CLAUDE.md →
"Verification — the regression anchor").

The cheapest anchor is balanced-parentheses accuracy (Table 3), e.g.:

```
GPT-2 Small : 100 / 100 / 49 / 0    (1/2/3/4-paren)
CodeLlama-7b: 99 / 100 / 98 / 87
Pythia-6.9b : 100 / 100 / 100 / 83
```

The anchor itself is captured on GPU as a separate step (not during scaffold).
After each experiment migration, compare the refactored output against the anchor
**for the same model set** — removing a debug `[:2]` can change which models run,
which must not be mistaken for a regression.
