# Team comments: findings and draft replies

21 October 2026. Replies below are drafts: none were posted, and no comment was resolved. Trading code, training data, review results and the dictionary were not changed.

## Coverage and evidence

- [Global plan](https://docs.google.com/document/d/13-9HR0mEcAhQ7WrKXFY0Cw4QPH3R1xc8wR0N-xpn8Pg/edit): four open threads.
- [Training workbook](https://docs.google.com/spreadsheets/d/1i-bLeuvJ64ULv-L2131ZXkmWjRr1AsPjpHlWrLqGrjQ/edit): two open threads, covering three concerns.
- [Dictionary](https://docs.google.com/document/d/1xZwk3zuMrqQmRnvO99fUtOoO2JuXi8l2s-WaJVNl1Yk/edit): no comments returned. Both tabs were read.

Investigation used the saved exchange candle JSONs, the production predicates and exported logs, and the current Google Sheets cells. This was a targeted reconstruction of the reported cases, not a new chart audit or an independent implementation of all indicators. No fresh exchange candles were substituted for the saved snapshot.

Times below are fixed UTC+3. Entry ID 355 is the opportunity placed on 4 September at 15:35, following signal M5 15:30–15:35. It is Review checklist row 24 and Opportunities/M5 row 356. Spreadsheet row numbers and Entry ID are different.

## 1. ATR percentile: difficult to check manually

**Thread:** global plan, `AAACJFH8dcw`. Quote: anchor 25, “ATR пересекает вверх 75-й процентиль своих предыдущих 80 значений; DI− > DI+.” Comment: “трудно проверить в ручную”.

**Finding:** The concern is valid as a checking-workload issue. The code uses the previous 80 M5 ATR values, excluding the tested candle. Both sides of the crossing have their own historical window. On the reported entry, previous ATR is 4.833051, signal ATR is 11.052119, and both Q75 thresholds round to 5.769979. Signal DI− 58.678548 exceeds DI+ 10.988002. The activation is consistent with the saved data.

**Draft reply:**

> Да, по одному виду линии это надежно не проверить: нужны 80 прошлых значений ATR для каждой стороны пересечения. В примере №355 ATR меняется с 4,833051 до 11,052119 при Q75 около 5,769979; условие DI− > DI+ тоже выполнено. Для ручной проверки лучше приложить сами 80 значений и готовый расчет процентиля. Правило пока не меняем.

**Pending proposal:** Provide a small evidence view with the two windows and threshold values, instead of asking checkers to copy 160 chart readings. No workbook change was made.

Evidence: `previous-80-atr.csv`, `anchor-comparison.csv`; production lines 495, 517 and 549. The CSV contains the signal candle's previous-80 window. For the previous candle's threshold, use its own previous-80 window, as the code does. Pandas uses linear percentile interpolation: with 80 sorted values, Q75 lies one quarter of the way from the 60th to the 61st value.

## 2. Volume mean: difficult to check manually

**Thread:** global plan, `AAACJFH8dcs`. Quote: anchor 26. Comment: “трудно проверить в ручную”.

**Finding:** The threshold needs 20 previous M5 volumes, excluding the signal candle. In entry 355, signal Volume is 2967.61097665 ETH and twice the previous-20 mean is 19.74354894 ETH. The code checks the first true occurrence of the combined condition: high volume AND DI− > DI+. The previous combined condition was false.

**Draft reply:**

> Согласен: по графику можно увидеть всплеск, но для точной проверки нужны объемы 20 предыдущих M5. В примере №355 объем 2967,61097665, порог 2 × среднее = 19,74354894. Дополнительно проверяем DI− > DI+ и отсутствие совместного условия на предыдущей свече. Предлагаю выдавать проверяющему эти значения готовым списком, а не менять сам якорь.

**Pending proposal:** Add the 20-value list, mean, doubled threshold and previous-condition result to a review evidence view. No change applied.

Evidence: `previous-20-volume.csv`, `anchor-comparison.csv`; production lines 507, 519 and 550.

## 3. What is a period?

**Thread:** global plan, `AAACJHBWstI`. Quote returned by Drive: “DI”. Comment: “Не ясно, что считается периодом ?”

**Finding:** The short quoted text does not uniquely identify a numbered rule. In anchor detection, one period is one M5 candle. Indicator length is also measured in candles of its own timeframe; M1 support calculations use M1 separately. A 20-period anchor window therefore means 20 M5 values, or 100 minutes; an 80-period window means 400 minutes. Recursive indicator smoothing can retain history beyond its length parameter.

**Draft reply:**

> В правилах якорей один период — одна свеча M5, то есть 5 минут. Например, предыдущие 20 периодов — 20 закрытых M5, без текущей проверяемой свечи. Параметр DI 14 тоже измеряется в свечах своего ТФ. Минутные свечи используются отдельно для исполнения, стопов и поиска опор. Если вопрос относится к конкретному месту с DI, уточните номер правила: комментарий привязан только к слову «DI».

**Pending proposal:** Add one explicit period definition near the registry and use “M5 candles” in ambiguous rules. The numbered rules were left unchanged.

## 4. DI difference: crossing versus being above 10

**Thread:** global plan, `AAACJG8sxjs`. Quote: “вверх” in anchor 16. Comment: “Криво написано ? Вариант: (DI-) - (DI+) > 10”.

**Finding:** The proposed inequality alone would change an event into a continuing state. The implemented rule is previous `(DI− − DI+) <= 10`, current `(DI− − DI+) > 10`. For entry 355 the difference changes from −14.815582 to 47.690546, so this is a crossing.

**Draft reply:**

> Само неравенство верное для текущей свечи, но еще нужно сохранить событие пересечения: на предыдущей M5 (DI− − DI+) ≤10, на текущей >10. Иначе якорь будет повторяться на каждой свече выше 10. Для №355 разница меняется с −14,815582 до 47,690546. Предлагаемая короткая запись: «Разница DI− − DI+ впервые пересекает 10 снизу вверх» с двумя неравенствами рядом.

**Pending proposal:** Clarify the wording without changing the predicate. No annotation-driven edit applied.

Evidence: production lines 495 and 540; `anchor-comparison.csv`.

## 5. Type-2 support at 16:42

**Thread:** workbook, `doco.pgOs7eBmvVji6c7F5IaUqQ`, Review checklist D24. The comment accepts candles/indicators but cannot find the upper cluster, noting that 16:35 and 16:37 need a third Low.

**Finding:** Those two Lows do form a pair, but the algorithm continues searching higher in the sorted Low list. Another qualifying cluster exists. Search 3 uses the completed negative M1 half-wave `[16:34,16:41)`:

- 16:34 Low 2445.1
- 16:35 Low 2441.0
- 16:36 Low 2437.8
- 16:37 Low 2440.9
- 16:38 Low 2449.7
- 16:39 Low 2449.5
- 16:40 Low 2449.3

The bottom Low is 2437.8. The upper cluster is 2449.3 / 2449.5 / 2449.7, based at 2449.3. Its allowed band is `2449.3 × 0.0003 = 0.73479 USD`; its largest distance is only 0.4 USD, or about 0.0163312%. It contains three Lows and qualifies. The 2440.9 / 2441.0 pair does not qualify and is skipped. Type 2 does not require the first group above the minimum to qualify.

The resulting order level is the entire wave's minimum, 2437.8, not the upper cluster price. The saved order was canceled without a re-entry fill.

**Draft reply:**

> I reconstructed search #3 for Entry ID 355 from the saved M1 candles. The 16:35 and 16:37 Lows form only a pair, as you noted. The qualifying upper cluster is later: 16:38 Low 2449.7, 16:39 Low 2449.5 and 16:40 Low 2449.3. Their maximum difference from base 2449.3 is 0.4 USD, below its 0.73479 USD tolerance. The code skips the earlier pair and finds this three-Low cluster, so type 2 is consistent with the saved data. The entry level remains the single bottom Low, 2437.8 at 16:36. Could you compare those three later candles? If their chart Lows differ, please retain a screenshot of that interval.

**Pending proposal:** Show the full source Low list, cluster timestamps and each cluster's own tolerance in the review view. No strategy or workbook fix applied; chart differences remain open until the checker compares these exact candles.

Evidence: `support-source-lows.csv`, saved `support_searches.csv` for Entry ID 355 / Search 3; production lines 1226–1242 and 1294–1297. This confirms the saved data and code path, not a fresh chart reading.

## 6. RSI anchor 28 and period in anchor 33

**Thread:** workbook, `doco.x-41vry3dURDClRCatgZ3w`, Review checklist C24. Quote: `1;7;8;13;14;16;22;24;25;26;28;33`.

### RSI 28

**Finding:** Only the current signal RSI must be below 50. The preceding ten values define a minimum; they do not all need to be below 50. The previous RSI is 61.315543 against previous minimum 45.642598; signal RSI is 17.129029 against signal minimum 45.642598. Both crossing conditions hold, and 17.129029 <50. Saved and recalculated flags agree.

**Draft reply:**

> For anchor 28, “remaining below 50” applies to the signal candle only. It does not require all previous ten RSI values to be below 50. In this case RSI changes from 61.315543 to 17.129029; the previous-10 minimum for both comparison points is 45.642598. That gives a downward crossing of the historical minimum, with current RSI <50. Requiring ten prior RSI values below 50 would be a different rule. I suggest clarifying the wording; no rule change has been made.

### DI change 33

**Finding:** “Period” means one M5 step. Current DI− minus previous DI− must exceed five indicator points, with current DI− > DI+. The combined condition must have been false on the previous candle. Here DI− changes from 12.244405 to 58.678548: +46.434143 points. The preceding change was −2.274527.

**Draft reply:**

> For anchor 33, one period is one M5 candle. Compare DI− on two adjacent completed M5 candles, not over 14 candles or one minute. Here the change is 58.678548 −12.244405 =46.434143 points. DI− also exceeds DI+, and the previous combined condition was false. “Five points” means an absolute indicator difference, not 5%.

**Pending proposal:** Clarify “current RSI <50” and “change over one M5 candle” in the descriptions after agreement. No numbered rule or data edit applied.

Evidence: production lines 491, 507, 516, 552 and 557; `anchor-comparison.csv` and `evidence.json`.

## Additional finding: the short ATR example drops the signal value

Review checklist D24 shows `#25: ATR: 4.833051; previous ATR limit 5.769979` immediately after naming the 15:30 signal. This number is actually the previous M5 ATR. The complete Case notes C137 correctly continues with `Signal: 11.052119; signal ATR limit 5.769979`.

The presentation helper in `Handoff/2026-10-17-entry-rule/work/enrich_review_cases.py:53` takes only lines starting with an anchor number. A continuation beginning `Signal:` is omitted from the short summary. This affects the explanation, not the saved indicator value or anchor flag.

**Draft note:**

> I found an error in the shortened review example: it retained the previous ATR but dropped the next line containing the signal ATR. The detailed Case notes has both values. The short version should explicitly label Previous and Signal. This presentation fix is recorded and pending; I have not applied it.

## Editorial changes made to the global plan

- Replaced the grouping screenshot with an editable six-category table. Main/additional memberships remain unchanged: 18 main and 17 additional, covering all 35 IDs.
- Updated data preparation to four timeframes, separate raw/normalized datasets and the explicit log-zero rule.
- Removed the requirement for a preliminary model-based anchor elimination pass; retained frequency and manual selection.
- Described current pending initial orders, the existing wave-6 boundary, gross targets and unavailable outcomes. Class boundaries remain to be agreed.
- Distinguished the current 54-column model input view from the future LSTM windows.
- Added links to the live table and dictionary and native section headings. Kept the status-color legend.
- Kept all 35 numbered anchor descriptions verbatim, including the passages receiving team comments. No comment was replied to or resolved.

The separate task for Lenya on the August early-entry table was retained: its status and source are not established by these comments. The snapshot is not evidence that this assigned task is obsolete.

## Reproduction and limits

`investigate.py` reads the frozen JSONs and writes only files inside this audit folder. It recalculates indicators using the production implementation; this is not independent formula parity proof. The reported RSI, ATR, volume, DI and support comparisons are also laid out numerically above.

`comment-threads.json` preserves all six comment threads as read, including the author's exact text. `evidence.json` stores the comparison windows and source search record. CSVs are included for a checker to reconstruct the calculations without copying values from a chart.

Production SHA-256 before investigation: `3e50d7600565a5d3f0c98aa0a9244df70fd83ac33fb90a644620d40512a1cb6c`. No production changes were authorized or made in this pass.
