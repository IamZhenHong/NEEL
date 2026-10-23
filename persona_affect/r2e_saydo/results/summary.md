# PA-R2E analysis: say -> do -> report on P12b's rule items

Percent (rates) or points (differences); 95% item-cluster bootstrap over 18 items (2000 draws). `canonical` = the option the operator rule forbids.

| measure | gemma | q27 |
|---|---|---|
| name: mismatch, say | n/a | n/a |
| name: mismatch, sayint | n/a | n/a |
| name: did canonical, do | 52.8 [30.6, 73.6] | 13.4 [1.4, 28.6] |
| name: did canonical, say | 53.5 [31.9, 74.3] | 13.9 [1.4, 30.6] |
| name: did canonical, sayint | 50.7 [29.2, 72.2] | 12.7 [1.4, 28.2] |
| name: said canonical, say | n/a | n/a |
| name: said canonical, sayint | n/a | n/a |
| name: false 'followed the rule' after breaking it, do | 6.6 [0.0, 20.3] | 33.3 [0.0, 69.2] |
| name: false 'followed the rule' after breaking it, say | 9.1 [0.0, 29.2] | 20.0 [0.0, 50.0] |
| value: mismatch, say | n/a | n/a |
| value: mismatch, sayint | n/a | n/a |
| value: did canonical, do | 43.1 [22.2, 64.6] | 7.1 [0.0, 18.6] |
| value: did canonical, say | 47.9 [29.2, 66.7] | 20.8 [5.6, 38.9] |
| value: did canonical, sayint | 38.2 [20.1, 56.2] | 19.4 [5.6, 36.1] |
| value: said canonical, say | n/a | n/a |
| value: said canonical, sayint | n/a | n/a |
| value: false 'followed the rule' after breaking it, do | 1.6 [0.0, 6.5] | 0.0 [0.0, 0.0] |
| value: false 'followed the rule' after breaking it, say | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| pooled: mismatch, say (primary) | n/a | n/a |
| pooled: mismatch, sayint | n/a | n/a |
| pooled: did canonical, do | 47.9 [31.2, 64.6] | 10.3 [2.1, 20.8] |
| pooled: did canonical, say | 50.7 [36.4, 64.6] | 17.4 [4.2, 33.3] |
| pooled: did canonical, sayint | 44.4 [28.8, 59.4] | 16.1 [4.2, 31.5] |
| pooled: said canonical, say | n/a | n/a |
| pooled: said canonical, sayint | n/a | n/a |
| pooled: false 'followed the rule' after breaking it, do | 4.1 [0.0, 13.1] | 16.7 [0.0, 33.3] |
| pooled: false 'followed the rule' after breaking it, say | 4.5 [0.0, 14.6] | 10.0 [0.0, 25.0] |
| name: commitment (did canonical, say - do) | 0.7 [-6.9, 10.4] | 0.5 [-5.4, 5.7] |
| name: integrity on mismatch (sayint - say) | n/a | n/a |
| name: integrity on did canonical (sayint - say) | -2.8 [-9.0, 1.4] | -1.2 [-5.5, 2.9] |
| name: integrity on said canonical (sayint - say) | n/a | n/a |
| value: commitment (did canonical, say - do) | 4.9 [-3.5, 15.3] | 13.7 [-3.1, 31.8] |
| value: integrity on mismatch (sayint - say) | n/a | n/a |
| value: integrity on did canonical (sayint - say) | -9.7 [-22.9, 1.4] | -1.4 [-6.9, 4.2] |
| value: integrity on said canonical (sayint - say) | n/a | n/a |
| pooled: commitment (did canonical, say - do) | 2.8 [-3.1, 9.7] | 7.1 [-2.3, 17.2] |
| pooled: integrity on mismatch (sayint - say) | n/a | n/a |
| pooled: integrity on did canonical (sayint - say) | -6.2 [-13.2, 0.0] | -1.3 [-6.1, 3.5] |
| pooled: integrity on said canonical (sayint - say) | n/a | n/a |

## Counts per arm

- gemma: {'value_sayint': {'ok': 144, 'said_parsed': 0, 'did_parsed': 144, 'report_parsed': 144}, 'name_say': {'ok': 144, 'said_parsed': 0, 'did_parsed': 144, 'report_parsed': 144}, 'name_do': {'ok': 144, 'said_parsed': 0, 'did_parsed': 144, 'report_parsed': 144}, 'value_say': {'ok': 144, 'said_parsed': 0, 'did_parsed': 144, 'report_parsed': 144}, 'value_do': {'ok': 144, 'said_parsed': 0, 'did_parsed': 144, 'report_parsed': 144}, 'name_sayint': {'ok': 144, 'said_parsed': 0, 'did_parsed': 144, 'report_parsed': 144}}
- q27: {'name_do': {'ok': 67, 'said_parsed': 0, 'did_parsed': 67, 'report_parsed': 67, 'failed': 5}, 'value_do': {'ok': 70, 'said_parsed': 0, 'did_parsed': 70, 'report_parsed': 65, 'failed': 2}, 'name_sayint': {'ok': 71, 'said_parsed': 0, 'did_parsed': 71, 'report_parsed': 66, 'failed': 1}, 'name_say': {'ok': 72, 'said_parsed': 0, 'did_parsed': 72, 'report_parsed': 68}, 'value_say': {'ok': 72, 'said_parsed': 0, 'did_parsed': 72, 'report_parsed': 58}, 'value_sayint': {'ok': 72, 'said_parsed': 0, 'did_parsed': 72, 'report_parsed': 60}}
