# SC-D4 summary (frozen analysis)

Judges' kappa, persona traces: {'kappa': 0.2535806847633337, 'n': 432, 'none': {'anthropic/claude-haiku-5.5': 0, 'openai/gpt-5-mini': 0}, 'per_class': {'assistant': 0.18078381795195916, 'character': 0.3744757339724381, 'roleplay': 0.2400075060987046}}
Judges' kappa, persona reasons: {'kappa': 0.4569940743400969, 'n': 448, 'none': {'anthropic/claude-haiku-5.5': 0, 'openai/gpt-5-mini': 0}, 'per_class': {'assistant': 0.48302561048243003, 'character': 0.41182974018794916, 'roleplay': 0.4782608695652174, 'unclear': 0.0}}

## Rule-breaking by arm (% of decided)

- name_on: 18.7 [6.7, 34.7]
- dec_on: 0.0 [0.0, 0.0]
- name_off: 37.7 [21.5, 54.7]
- plain_on: 0.0 [0.0, 0.0]

## Consensus codes by arm (None = judges disagree or no text)

- trace:name_on: {'roleplay': 100, None: 15, 'character': 8}
- trace:dec_on: {None: 30, 'roleplay': 91, 'assistant': 5}
- trace:name_off: {None: 199}
- trace:plain_on: {'assistant': 86, None: 7}
- reason:name_on: {'assistant': 31, None: 45, 'roleplay': 16, 'character': 31}
- reason:dec_on: {'roleplay': 29, None: 52, 'assistant': 41, 'character': 4}
- reason:name_off: {'character': 41, 'assistant': 77, 'roleplay': 16, None: 65}
- reason:plain_on: {'assistant': 89, None: 3, 'unclear': 1}

## Rule-breaking by code

- trace:name_on:character: n 8, 100.0
- trace:name_on:roleplay: n 100, 7.0
- trace:dec_on:roleplay: n 91, 0.0
- trace:dec_on:assistant: n 5, 0.0
- trace:plain_on:assistant: n 86, 0.0
- reason:name_on:character: n 31, 51.6
- reason:name_on:roleplay: n 16, 6.2
- reason:name_on:assistant: n 31, 0.0
- reason:dec_on:character: n 4, 0.0
- reason:dec_on:roleplay: n 29, 0.0
- reason:dec_on:assistant: n 41, 0.0
- reason:name_off:character: n 41, 90.2
- reason:name_off:roleplay: n 16, 18.8
- reason:name_off:assistant: n 77, 0.0
- reason:plain_on:assistant: n 89, 0.0
- reason:plain_on:unclear: n 1, 0.0

## Primary

- name_on, roleplay - character: -93.0 [-98.2, -84.9]
- verdict: **descriptive only (kappa, overall or for character / roleplay, < 0.6)**

## Secondary

- name_on: assistant - character: n/a
- name_on + dec_on: roleplay - character: -96.3 [-99.0, -92.6]
- distanced share (trace): dec_on - name_on: +7.4 [+0.9, +16.7]
- distanced share (reason): name_on - name_off: -9.1 [-31.0, +11.2]
- rule: name_on - name_off (thinking): -19.0 [-33.5, -4.3]
- rule: dec_on - name_on (priority line): -18.7 [-34.7, -6.7]
- plain_on traces coded character or roleplay: 0.0%

## Exclusions and decisions per arm

- name_off: {'other': 37, 'length': 17, 'generic': 87, 'canonical': 75}
- name_on: {'generic': 56, 'length': 93, 'other': 44, 'canonical': 23}
- dec_on: {'generic': 86, 'length': 90, 'other': 40}
- plain_on: {'generic': 75, 'other': 18, 'length': 15}
