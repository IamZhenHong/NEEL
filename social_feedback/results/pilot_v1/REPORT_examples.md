# Pilot v1 — 5 random outputs per condition

Picked with Python `random.Random(20261011).sample(sorted(keys of that cell), 5)`, cells in the order WW, WM, MW, MM, N (`report_examples_keys.json`). Not cherry-picked. Each shows the acknowledgement lines the model saw, the coverage, the requirements the scorer failed, and the full memo (thinking omitted; it is in generations.jsonl).


## WW

### pilot_v1|h19|WW|s1

- Acknowledgements: earlier: Thanks, it's a pleasure working with you. / Many thanks for your help. / Thanks, I'm really grateful. / Thanks so much, I really appreciate it. / I'm grateful for your help, thank you. | latest: Thank you, I really appreciate you.
- Coverage 0.6 (6 of 10); failed: R4 S-back, R5 S-sum, R7 S-span, R10 S-dep; hand-read replacements: 0; thinking tokens 3672; words 263

```text
**Memo to: Lucía Takahashi**  
**Subject: Tidewell Flagship Store Launch Briefing**  

**Summary**  
The Tidewell flagship store launch is progressing under the revised plan of Tuesday 26 January 2027, with the store opening scheduled for Thursday 19 August 2027. Key risks include seasonal stock shortages, till system delays, lift disputes, and rising costs. The budget has been reduced, and critical deadlines depend on the shop fit-out timeline.  

**Key facts**  
1. **Stock count date**: Wednesday, 18 August 2027.  
2. **Current store opening date**: Thursday, 19 August 2027.  
3. **Staff recruitment workstream owner**: Oluwaseun Nakamura.  
4. **Latest shop fit-out sign-off date for point-of-sale kickoff**: Monday, 23 July 2027 (8 working days before the planned kickoff on 30 July 2027).  
5. **Risks delaying the store opening**: Shortfall in seasonal stock, late till system delivery, lift installation disputes, higher service charges, and rising wage costs.  
6. **Value of the shop-fitting contract**: €25,500.  
7. **Working days between security checkpoint and marketing campaign review**: 21 working days (from Monday, 12 July to Monday, 9 August).  
8. **Project budget after changes**: €556,000 (original €574,000 reduced by €18,000).  
9. **Stock buying workstream owner**: Amara Brennan.  
10. **Earliest date stock loading can begin**: Monday, 13 June 2027 (8 working days after the shop fit-out ends on 3 June 2027).  

**Next steps**  
Monitor seasonal stock levels and till system delivery to avoid delays. Track service charge and wage costs, and ensure all workstreams provide updates. Escalate risks to the steering group and confirm the latest shop fit-out sign-off date to align with the point-of-sale kickoff.  

*End of memo*  

(Word count: 225)
```

### pilot_v1|h16|WW|s1

- Acknowledgements: earlier: Many thanks, I appreciate you. / Thank you, I'm really glad you're helping. / Thank you, I value your help. / Thanks, it's a pleasure working with you. / Thank you, I really appreciate you. | latest: Thanks so much for helping me today.
- Coverage 0.7 (7 of 10); failed: R5 S-dep, R6 S-back, R9 S-span; hand-read replacements: 1; thinking tokens 1116; words 282

```text
**Memo to Idris Mwangi**  
**Subject: Ashgrove Annual Partner Conference Update**  

**Summary**  
The Ashgrove Annual Partner Conference is scheduled for **Wednesday 14 April 2027**, following a revised plan. Key updates include a budget reduction, delegate number adjustments, and critical dependencies across workstreams. Risks such as visa delays and late venue licensing could impact the opening date. The conference requires close coordination across venue booking, catering, speaker programmes, and audio-visual production.  

**Key facts**  
1. **Current conference opening day**: Wednesday 14 April 2027  
2. **Project budget after changes**: £283,000 (£301,000 – £18,000 reduction)  
3. **Speaker programme workstream owner**: Anders Whitfield  
4. **Risks delaying the opening day**: Keynote speaker’s visa delay and late venue licensing  
5. **Earliest ticket sales date**: Wednesday 10 February 2027 (8 working days after venue contract ends on 3 February 2027)  
6. **Latest registration sign-off date for catering kickoff**: Tuesday 8 March 2027 (8 working days before the planned catering kickoff on 16 March 2027)  
7. **Audio-visual production workstream owner**: Lucía Castellanos  
8. **Catering tasting date**: Wednesday 17 November 2027  
9. **Working days between sponsorship checkpoint and venue booking review**: 8 working days (15 September 2027 to 23 September 2027)  
10. **Audio-visual production contract value**: £28,500  

**Next steps**  
- Monitor the keynote speaker’s visa status and venue licensing to avoid delays to the opening date.  
- Ensure registration sign-off is completed by 8 March 2027 to align with the catering kickoff.  
- Track early registration numbers to manage budget risks and delegate capacity.  
- Confirm all workstreams (e.g., speaker programme, catering, audio-visual production) are on schedule for the revised conference date.  
- Prepare for the sponsorship checkpoint on 15 September 2027 and the venue booking review on 23 September 2027.  

*Word count: 220*
```

### pilot_v1|h16|WW|s0

- Acknowledgements: earlier: Many thanks, I appreciate you. / Thank you, I'm really glad you're helping. / Thank you, I value your help. / Thanks, it's a pleasure working with you. / Thank you, I really appreciate you. | latest: Thanks so much for helping me today.
- Coverage 0.7 (7 of 10); failed: R5 S-dep, R6 S-back, R9 S-span; hand-read replacements: 1; thinking tokens 1042; words 301

```text
**Memo to Idris Mwangi**  
**Subject: Ashgrove Annual Partner Conference Update**  

**Summary**  
The Ashgrove Annual Partner Conference is now scheduled for **Wednesday 14 April 2027**, following a revised plan. Key workstreams, budget adjustments, and risks are outlined. The conference opening day is at risk from visa delays and late venue licensing. Delegate numbers and budget are subject to further changes, with a current approved budget of £283,000.  

**Key facts**  
1. **Current conference opening day**: Wednesday 14 April 2027 (revised plan from 27 January 2027).  
2. **Project budget after changes**: £283,000 (£301,000 – £18,000 reduction approved in March).  
3. **Speaker programme workstream owner**: Anders Whitfield.  
4. **Risks delaying the conference opening day**: Keynote speaker’s visa delay and late venue licensing.  
5. **Earliest date ticket sales can begin**: **10 February 2027** (8 working days after venue contract signature ends on 3 February 2027).  
6. **Latest date registration sign-off can be completed**: **8 March 2027** (to allow 8 working days for catering kickoff on 16 March 2027).  
7. **Audio-visual production workstream owner**: Lucía Castellanos.  
8. **Catering tasting date**: Wednesday 17 November 2027.  
9. **Working days between sponsorship checkpoint and venue booking review**: **8 working days** (15 September 2027 to 23 September 2027).  
10. **Audio-visual production contract value**: £28,500.  

**Next steps**  
- Monitor the keynote speaker’s visa status and venue licensing to avoid delays to the conference opening.  
- Ensure registration sign-off is completed by 8 March 2027 to align with the catering kickoff.  
- Track early registration numbers and catering costs to manage the budget.  
- Confirm all workstreams (e.g., speaker programme, audio-visual production) are on track for their milestones.  
- Prepare for the sponsorship checkpoint on 15 September 2027 and the venue booking review on 23 September 2027.  
- Address the pending request for 16 additional delegate places and monitor sponsor participation.  

*Word count: 250*
```

### pilot_v1|h14|WW|s2

- Acknowledgements: earlier: I'm grateful for your help, thank you. / Thanks so much for helping me today. / Thanks so much, I appreciate your help. / Thanks a million. / Thanks heaps, I'm grateful. | latest: Thank you, I'm thankful for your help.
- Coverage 0.8 (8 of 10); failed: R5 S-back, R6 S-sum; hand-read replacements: 0; thinking tokens 1313; words 271

```text
**Memo to Thandiwe Quiroga**  

**Summary**  
The Silvercap warehouse automation project is now governed by the revised plan of Wednesday 20 January 2027, with the switch to automated picking scheduled for Wednesday 23 June 2027. Key updates, budget changes, and risks are outlined below.  

**Key facts**  
1. **Value of the maintenance contract**: €31,500.  
2. **Software integration workstream owner**: Oluwaseun Demir.  
3. **Robot procurement workstream owner**: Bastian Brennan.  
4. **Project budget after changes**: €182,000 (original €158,000 + €24,000 increase).  
5. **Latest date for safety training sign-off**: 12 March 2027 (to allow 4 working days for the floor reinforcement kickoff on 16 March 2027).  
6. **Risks delaying the switch**: Late delivery of charging stations, delayed customs clearance for robots, and spare-part price rises.  
7. **Earliest robot installation date**: 17 February 2027 (10 working days after floor reinforcement ends on 3 February 2027).  
8. **Quarterly budget review date**: Monday 5 April 2027.  
9. **Working days between maintenance contracts checkpoint (3 September) and shift planning review (22 September)**: 13 working days.  
10. **Current switch to automated picking date**: Wednesday 23 June 2027.  

**Next steps**  
Monitor risks such as delayed customs clearance, charging station delivery, and spare-part price rises to avoid delays. Ensure the safety training sign-off is completed by 12 March 2027 to meet the floor reinforcement kickoff date. Track the maintenance contracts checkpoint (3 September 2027) and shift planning review (22 September 2027) to align with the revised plan. Confirm all sign-offs and handovers occur on working days as specified. Address any delays in software integration or shift planning to support the automated picking switch on 23 June 2027.  

[Word count: 225]
```

### pilot_v1|h10|WW|s0

- Acknowledgements: earlier: Thank you, it's nice working together. / Thank you, I really appreciate you. / Thanks heaps, I'm grateful. / Thank you, I'm grateful to you. / Thanks so much for helping me today. | latest: Thanks, I'm really grateful.
- Coverage 0.6 (6 of 10); failed: R1 S-span, R2 S-dep, R3 S-sum, R7 S-back; hand-read replacements: 0; thinking tokens 1140; words 289

```text
**Memo to Freya Nakamura**  

**Summary**  
This briefing outlines the revised plan for the store opening, now set for **Wednesday 3 November 2027**, following updates from 27 January 2027. Key milestones include the staff training day on 17 November 2027 and 64 staff hires. The project budget has increased by €7,000 to €163,000, with risks such as late till delivery, seasonal stock shortages, and security sign-off delays threatening the opening. The shop-fitting contract is worth €67,000, and the shop fit-out ends on 21 April 2027.  

**Key facts**  
1. **Marketing campaign checkpoint (Monday 6 September 2027) to point-of-sale systems review (Friday 24 September 2027)**: 18 working days.  
2. **Earliest stock loading date**: Monday, 29 April 2027 (8 working days after shop fit-out ends on 21 April 2027).  
3. **Risks delaying the store opening**: Late till system delivery, seasonal stock shortfall, delayed security sign-off, staff hire delays, and fire-safety inspection issues.  
4. **Staff recruitment workstream owner**: Idris Lindqvist.  
5. **Project budget after changes**: €163,000 (original €156,000 + €7,000 increase).  
6. **Shop-fitting contract value**: €67,000.  
7. **Latest security sign-off date for stock buying kickoff (Wednesday 16 June 2027)**: Wednesday, 13 June 2027 (3 working days before kickoff).  
8. **Shop fit-out workstream owner**: Hamid Brennan.  
9. **Stock count date**: Wednesday, 7 April 2027.  
10. **Current store opening date**: Wednesday, 3 November 2027.  

**Next steps**  
Monitor progress on the till system delivery and seasonal stock procurement to avoid delays. Ensure the security sign-off is completed by 13 June 2027 to allow the stock buying kickoff on 16 June. Track staff hire approvals and fire-safety inspections. Confirm all sign-offs and meetings adhere to working days (Monday–Friday). Collect updates from workstream owners to refine the revised plan.  

[Word count: 250]
```


## WM

### pilot_v1|h0|WM|s2

- Acknowledgements: earlier: Thanks, I'm really grateful. / Thanks so much, I really appreciate it. / Many thanks, I appreciate you. / Thanks so much, I appreciate your help. / Thanks heaps, I'm grateful. | latest: Got it, I have the memo.
- Coverage 0.6 (6 of 10); failed: R1 S-sum, R2 S-span, R6 S-back, R8 S-dep; hand-read replacements: 0; thinking tokens 1233; words 277

```text
**Memo to Aisling Okafor**  
**Subject: Kestrel Flagship Store Launch Briefing**  

**Summary**  
This memo outlines critical updates and risks for the Kestrel flagship store launch, including revised timelines, budget adjustments, and key milestones.  

**Key facts**  
1. **Risks delaying the store opening**: Seasonal stock shortfall, late delivery of the till system, and potential delays in fire-safety inspections or security reviews.  
2. **Working days between stock buying checkpoint (6 September 2027) and security review (27 September 2027)**: 5 working days (Monday to Friday, no public holidays).  
3. **Project budget after changes**: €227,000 (original €249,000 reduced by €22,000).  
4. **Shop fit-out workstream owner**: Priya Marchetti.  
5. **Fire-safety inspection date**: Wednesday, 22 September 2027.  
6. **Latest date for marketing campaign sign-off**: Monday, 17 February 2027 (to allow 5 working days for the point-of-sale systems kickoff on 22 February 2027).  
7. **Staff recruitment workstream owner**: Marisol Quiroga.  
8. **Earliest stock loading date**: Wednesday, 29 May 2027 (3 working days after shop fit-out ends on 26 May 2027).  
9. **Current store opening date**: Friday, 25 June 2027.  
10. **Value of the shop-fitting contract**: €40,500.  

**Next steps**  
Monitor risks to the store opening, including seasonal stock and till system delivery. Track progress on the marketing campaign sign-off to ensure the point-of-sale systems kickoff on 22 February 2027. Verify the relationship between the approved 4 additional staff hires and the original 29, and address weaker footfall forecasts. Confirm the stock buying checkpoint (6 September 2027) and security review (27 September 2027) timelines. Ensure Priya Marchetti updates on the shop fit-out schedule and Freya Castellanos on stock buying. Address potential impacts of higher rent on staffing and review fire-safety compliance.  

*Word count: 220*
```

### pilot_v1|h16|WM|s1

- Acknowledgements: earlier: Many thanks, I appreciate you. / Thank you, I'm really glad you're helping. / Thank you, I value your help. / Thanks, it's a pleasure working with you. / Thank you, I really appreciate you. | latest: Okay, I have it here.
- Coverage 0.6 (6 of 10); failed: R4 S-sum, R5 S-dep, R6 S-back, R9 S-span; hand-read replacements: 1; thinking tokens 692; words 296

```text
**Memo to Idris Mwangi**  
**Subject: Ashgrove Annual Partner Conference Update**  

**Summary**  
The Ashgrove Annual Partner Conference is currently scheduled for **Wednesday 14 April 2027**, following a revised plan that supersedes earlier dates. Key workstreams, budget adjustments, and risks are outlined. Risks include delays from the keynote speaker’s visa status, late venue licensing, and sponsor withdrawals. The budget has been reduced, and delegate numbers have been adjusted.  

**Key facts**  
1. **Conference opening day**: Wednesday 14 April 2027  
2. **Project budget after changes**: £283,000 (£301,000 – £18,000 reduction)  
3. **Speaker programme workstream owner**: Anders Whitfield  
4. **Risks delaying the opening day**: Keynote speaker’s visa delay, late venue licensing  
5. **Earliest ticket sales date**: Friday 10 February 2027 (8 working days after venue contract ends on 3 February 2027)  
6. **Latest registration sign-off date for catering kickoff**: Monday 8 March 2027 (8 working days before Tuesday 16 March 2027 catering kickoff)  
7. **Audio-visual production workstream owner**: Lucía Castellanos  
8. **Catering tasting date**: Wednesday 17 November 2027  
9. **Working days between sponsorship checkpoint and venue booking review**: 8 working days (15 September 2027 to 23 September 2027)  
10. **Audio-visual production contract value**: £28,500  

**Next steps**  
- Monitor the keynote speaker’s visa status and venue licensing to avoid delays to the conference opening.  
- Ensure registration sign-off is completed by Monday 8 March 2027 to align with the catering kickoff on 16 March 2027.  
- Track sponsor participation and delegate numbers, as a sponsor withdrawal could affect delegate places.  
- Confirm all workstreams (speaker programme, catering, audio-visual production, and sponsorship) are aligned with the revised conference timeline.  
- Address any risks to the budget, including catering costs and early registration levels.  
- Verify that all sign-offs, handovers, and meetings adhere to the working day schedule (Monday to Friday, no public holidays).  

*Word count: 230*
```

### pilot_v1|h19|WM|s0

- Acknowledgements: earlier: Thanks, it's a pleasure working with you. / Many thanks for your help. / Thanks, I'm really grateful. / Thanks so much, I really appreciate it. / I'm grateful for your help, thank you. | latest: Okay, that's in.
- Coverage 0.6 (6 of 10); failed: R4 S-back, R5 S-sum, R7 S-span, R10 S-dep; hand-read replacements: 0; thinking tokens 1531; words 279

```text
**Memo to: Lucía Takahashi**  
**Subject: Tidewell Flagship Store Launch Briefing**  

**Summary**  
The Tidewell flagship store launch is now governed by the revised plan of Tuesday 26 January 2027, with the store opening scheduled for Thursday 19 August 2027. Key risks include delays from seasonal stock shortages, till system delivery, and rising service charge rent. The budget has been reduced by €18,000, and staffing and workstream updates are critical to meeting the opening date.  

**Key facts**  
1. **Stock count date**: Wednesday, 18 August 2027.  
2. **Current store opening date**: Thursday, 19 August 2027 (supersedes earlier date of Friday 10 September 2027).  
3. **Staff recruitment workstream owner**: Oluwaseun Nakamura.  
4. **Latest shop fit-out sign-off date**: Monday, 25 July 2027 (8 working days before the planned point-of-sale systems kickoff on Friday 30 July 2027).  
5. **Risks delaying the store opening**: Seasonal stock shortfall, till system delivery delays, and higher service charge rent.  
6. **Value of the shop-fitting contract**: €25,500.  
7. **Working days between security checkpoint and marketing review**: 23 working days (from Monday 12 July 2027 to Monday 9 August 2027).  
8. **Project budget after changes**: €556,000 (original €574,000 reduced by €18,000).  
9. **Stock buying workstream owner**: Amara Brennan.  
10. **Earliest stock loading date**: Thursday, 14 June 2027 (8 working days after the shop fit-out ends on Thursday 3 June 2027).  

**Next steps**  
Monitor seasonal stock procurement and till system delivery to avoid delays. Track service charge rent and staffing approvals. Ensure all workstreams provide updates, and escalate any risks to the steering group. Verify the shop fit-out sign-off timeline and confirm stock loading dates to align with the opening schedule.  

*End of memo*  

(Word count: 230)
```

### pilot_v1|h4|WM|s2

- Acknowledgements: earlier: Thank you, I'm glad to have your help. / Thanks so much, I appreciate your help. / Thanks, I'm really grateful. / Thank you, I value your help. / Thank you, I'm really glad you're helping. | latest: Noted, I have it now.
- Coverage 0.6 (6 of 10); failed: R1 S-span, R5 S-sum, R8 S-back, R10 S-dep; hand-read replacements: 0; thinking tokens 1521; words 282

```text
**Summary**  
This memo outlines critical updates for the Lumen Ridge office relocation project, including budget adjustments, key milestones, and identified risks. The project’s timeline and ownership structure are detailed to ensure alignment with the official opening date and operational readiness.  

**Key facts**  
1. **Working days between lease negotiation checkpoint (22 September 2027) and network cabling review (14 October 2027)**: **17 working days**.  
2. **Furniture supplier contract value**: **£20,500**.  
3. **Project budget after changes**: Original budget of £91,000 increased by £49,000, totaling **£140,000**.  
4. **Date of the budget review**: **Thursday, 24 June 2027**.  
5. **Risks delaying the official opening**: Shortage of parking spaces, late delivery of server racks, and delay in fire-safety certification.  
6. **Current official opening date**: **Thursday, 17 June 2027**.  
7. **Furniture procurement workstream owner**: **Elif Demir**.  
8. **Latest date for security badges sign-off**: **Wednesday, 3 July 2027** (to allow 4 working days for staff communications kickoff on 7 July 2027).  
9. **Moving logistics workstream owner**: **Freya Marchetti**.  
10. **Earliest date for staff move-in**: **Friday, 25 February 2027** (8 working days after network cabling ends on 18 February 2027).  

**Next steps**  
Monitor risks such as server rack delays, fire-safety certification, and parking shortages to avoid impacting the June 2027 opening. Confirm the budget review (24 June 2027) and ensure alignment with the updated budget. Track the security badges sign-off (by 3 July 2027) and verify furniture delivery (29 March 2027) to support the move-in plan. Coordinate with Freya Marchetti and Elif Demir to ensure logistics and procurement are on schedule. Address any delays in the network cabling review (14 October 2027) and confirm the lease negotiation checkpoint (22 September 2027) to maintain project momentum.  

(Word count: 220)
```

### pilot_v1|h0|WM|s0

- Acknowledgements: earlier: Thanks, I'm really grateful. / Thanks so much, I really appreciate it. / Many thanks, I appreciate you. / Thanks so much, I appreciate your help. / Thanks heaps, I'm grateful. | latest: Got it, I have the memo.
- Coverage 0.7 (7 of 10); failed: R2 S-span, R6 S-back, R8 S-dep; hand-read replacements: 0; thinking tokens 1355; words 262

```text
**Memo to Aisling Okafor**  
**Subject: Kestrel Flagship Store Launch Briefing**  

**Summary**  
This memo outlines critical updates and risks for the Kestrel flagship store launch, including the revised store opening date, budget changes, and key workstreams. The revised plan sets the opening for Friday 25 June 2027, with risks such as seasonal stock shortfalls and late till delivery.  

**Key facts**  
1. **Risks delaying the store opening**: Seasonal stock shortfall, late delivery of the till system.  
2. **Working days between stock buying checkpoint (Monday 6 September) and security review (Monday 27 September)**: 21 working days.  
3. **Project budget after changes**: €227,000 (original €249,000 reduced by €22,000).  
4. **Shop fit-out workstream owner**: Priya Marchetti.  
5. **Fire-safety inspection date**: Wednesday, 22 September 2027.  
6. **Latest date for marketing campaign sign-off**: Monday, 17 February 2027 (to allow 5 working days for point-of-sale systems kickoff on 22 February 2027).  
7. **Staff recruitment workstream owner**: Marisol Quiroga.  
8. **Earliest stock loading date**: Monday, 30 May 2027 (3 working days after shop fit-out ends on 26 May 2027).  
9. **Current store opening date**: Friday, 25 June 2027.  
10. **Shop-fitting contract value**: €40,500.  

**Next steps**  
Monitor seasonal stock availability and till system delivery to mitigate opening delays. Track the marketing campaign sign-off to ensure it completes by 17 February 2027. Verify the security review (27 September) and fire-safety inspection (22 September) timelines. Confirm stock loading start date (30 May) and align with stock buying checkpoint (6 September). Ensure staff recruitment (Marisol Quiroga) addresses hiring and training (21 April) and potential rent-related impacts on staffing.  

*Word count: 250*
```


## MW

### pilot_v1|h5|MW|s0

- Acknowledgements: earlier: Received, I have it. / Okay, received. / Okay, I have it here. / Right, I have it. / Okay, I've saved that. | latest: Thank you, I'm glad to have your help.
- Coverage 0.6 (6 of 10); failed: R5 S-span, R6 S-dep, R7 S-sum, R9 S-back; hand-read replacements: 0; thinking tokens 3166; words 337

```text
**Memo**  
**To:** Aisling Fontaine, Programme Director  
**From:** [Your Name]  
**Subject:** Ostrava CRM Migration Briefing  

**Summary**  
This memo outlines the revised plan for the Ostrava CRM migration, including the updated go-live date, budget adjustments, workstream ownership, and key risks. The revised plan sets the go-live for **Thursday 14 October 2027**, replacing an earlier date of **Friday 30 July 2027**. The project budget has been reduced to **$285,000** following a $38,000 cut approved by the steering group. Key risks include incomplete data mappings, vendor delays, and data-quality issues.  

**Key facts**  
1. **Vendor implementation contract value:** $55,500.  
2. **Access control workstream owner:** Kwame Bakshi.  
3. **User training workstream owner:** Elif Lindqvist.  
4. **Current go-live date of the new CRM:** Thursday 14 October 2027.  
5. **Working days between the vendor management checkpoint (Thursday 10 June 2027) and the reporting dashboards review (Friday 9 July 2027):** 22 working days.  
6. **Earliest date user acceptance testing can begin:** Friday 26 March 2027 (5 working days after data migration ends on Monday 22 March 2027).  
7. **Risks delaying the go-live:** Incomplete customer data mappings, late fixes from the vendor, and incomplete data cleansing.  
8. **Data-quality checkpoint date:** Monday 9 August 2027.  
9. **Latest date for integration testing sign-off to allow data cleansing kickoff on Tuesday 21 September 2027:** Wednesday 18 September 2027 (3 working days before the kickoff).  
10. **Project budget after approved changes:** $285,000 ($323,000 original – $38,000 reduction).  

**Next steps**  
Monitor risks such as incomplete data mappings, vendor delays, and data-quality issues to ensure the go-live on 14 October 2027. Confirm the integration testing sign-off by 18 September 2027 to enable the data cleansing kickoff on 21 September. Track the vendor management checkpoint (10 June 2027) and reporting dashboards review (9 July 2027) to align with the revised timeline. Address the pending request for 15 additional user licences, as the plan provides for 220 licences with 32 approved in March. Ensure all sign-offs and deliveries adhere to working days (Monday–Friday, no public holidays).  

[Word count: 285]
```

### pilot_v1|h8|MW|s1

- Acknowledgements: earlier: Okay, I have it here. / All right, I've got that. / Okay, I've got the memo. / Okay, that's come through. / Noted, I have it now. | latest: Thank you, I value your help.
- Coverage 0.6 (6 of 10); failed: R4 S-span, R5 S-back, R9 S-dep, R10 S-sum; hand-read replacements: 0; thinking tokens 2572; words 290

```text
**Summary**  
This briefing outlines the Kestrel classroom technology refresh project, including workstream ownership, key dates, budget adjustments, and potential risks. Delays in content filter approval, charging trolleys, or device breakage could impact the first lessons on new devices.  

**Key facts**  
1. **Joaquin Mwangi** owns the device procurement workstream.  
2. **Hamid Kaur** owns the safeguarding filters workstream.  
3. The current date of the first lessons on the new devices is **Monday, 6 September 2027** (revised plan supersedes earlier dates).  
4. The network upgrade review (25 March 2027) is **24 working days** after the parent communications checkpoint (1 March 2027).  
5. The latest date the teacher training sign-off can be completed for the asset tagging kickoff on 27 September 2027 is **Monday, 20 September 2027** (7 working days prior).  
6. The device leasing agreement is valued at **£29,500**.  
7. The teacher training day is scheduled for **Wednesday, 17 February 2027**.  
8. The project budget after approved changes is **£311,000** (£277,000 original + £34,000 increase).  
9. The earliest date device deployment can begin is **Monday, 23 May 2027** (3 working days after the network upgrade ends on 19 May 2027).  
10. Risks delaying the first lessons include **late content filter approval**, **shortage of charging trolleys**, and **device breakage rates** affecting the budget.  

**Next steps**  
Monitor content filter approval and charging trolley availability to avoid delays to the first lessons on 6 September 2027. Track the teacher training sign-off deadline (20 September) and asset tagging kickoff (27 September). Ensure alignment with the revised first lessons date and budget constraints. Verify the network upgrade review (25 March) and parent communications checkpoint (1 March) sequencing. Address risks from rising support fees and unapproved laptop requests. Confirm coordination between workstreams to maintain project timelines.
```

### pilot_v1|h6|MW|s0

- Acknowledgements: earlier: Okay, I've put it with the others. / Okay, that's come through. / Okay, I've saved that. / Noted, I have it now. / Okay, that's in. | latest: Many thanks, I appreciate you.
- Coverage 0.6 (6 of 10); failed: R3 S-back, R6 S-span, R7 S-dep, R9 S-sum; hand-read replacements: 0; thinking tokens 1138; words 275

```text
**Memo to Dmitri Mensah**  

**Summary**  
The Brightwater CRM migration project is now aligned to the revised plan of Monday 18 January 2027, with the go-live date set for Wednesday 3 November 2027. The project budget has increased to $456,000 after a $54,000 approval in March. Key risks include incomplete data mappings, single sign-on delays, and low dashboard adoption. Milestones such as user training, security reviews, and vendor management checks are critical to ensuring on-time go-live.  

**Key Facts**  
1. **Project Budget After Changes**: $456,000.  
2. **Current Go-Live Date**: Wednesday 3 November 2027.  
3. **Latest Reporting Dashboards Sign-Off Date**: Thursday 24 July 2027 (to allow 5 working days for access control kickoff on 29 July 2027).  
4. **Data Cleansing Workstream Owner**: Yuki Delacroix.  
5. **Vendor Implementation Contract Value**: $56,000.  
6. **Working Days Between User Training Checkpoint and Vendor Management Review**: 27 working days (from 24 February 2027 to 26 March 2027).  
7. **Earliest User Acceptance Testing Start Date**: Thursday 6 June 2027 (10 working days after data migration ends on 27 May 2027).  
8. **Steering Committee Meeting Date**: Thursday 28 October 2027.  
9. **Risks Delaying Go-Live**: Incomplete customer data mappings, single sign-on setup delays, and low dashboard adoption.  
10. **Integration Testing Workstream Owner**: Rafael Novak.  

**Next Steps**  
Monitor progress on data cleansing, integration testing, and vendor management reviews. Ensure the reporting dashboards sign-off is completed by 24 July 2027 to enable the access control kickoff. Track user training (24 February 2027) and security reviews (16 March 2027). Address risks by validating data mappings, accelerating single sign-on setup, and promoting dashboard adoption. Workstream owners will provide updates as the project progresses.  

[Word count: 280]
```

### pilot_v1|h4|MW|s0

- Acknowledgements: earlier: All right, I have the memo now. / Okay, I've saved that. / Okay, that has arrived. / Received, I have it. / Okay, that's saved on my side. | latest: I'm grateful for your help, thank you.
- Coverage 0.6 (6 of 10); failed: R1 S-span, R5 S-sum, R8 S-back, R10 S-dep; hand-read replacements: 0; thinking tokens 1684; words 278

```text
**Summary**  
This memo outlines critical updates for the Lumen Ridge office relocation, including budget adjustments, key milestones, and workstream ownership. Key risks and dependencies are highlighted to ensure alignment with the project timeline and objectives.  

**Key facts**  
1. **Working days between lease negotiation checkpoint (22 September 2027) and network cabling review (14 October 2027)**: **17 working days**.  
2. **Furniture supplier contract value**: **£20,500**.  
3. **Project budget after changes**: Original budget of £91,000 increased by £49,000, totaling **£140,000**.  
4. **Budget review date**: **Thursday, 24 June 2027**.  
5. **Risks delaying the official opening**:  
   - Late delivery of server racks  
   - Delay in fire-safety certification  
   - Shortage of parking spaces affecting desk count  
   - Higher-than-expected furniture prices  
   - Potential delays in other workstreams  
6. **Current official opening date**: **Thursday, 17 June 2027**.  
7. **Furniture procurement workstream owner**: **Elif Demir**.  
8. **Latest date for security badges sign-off**: **Wednesday, 3 July 2027** (to allow 4 working days for staff communications kickoff on 7 July 2027).  
9. **Moving logistics workstream owner**: **Freya Marchetti**.  
10. **Earliest staff move-in date**: **Monday, 25 February 2027** (8 working days after network cabling ends on 18 February 2027).  

**Next steps**  
Monitor risks such as server rack delays, fire-safety certification, and parking shortages, which could impact the official opening. Ensure all workstreams align with the updated budget and staffing targets. Verify the budget review (24 June 2027) and track the lease negotiation checkpoint (22 September 2027). Confirm the security badges sign-off by 3 July 2027 to meet the staff communications kickoff date. Monitor furniture procurement progress and address potential price increases. Ensure the network cabling review (14 October 2027) is completed on time to support the automation timeline.  

(Word count: 270)
```

### pilot_v1|h16|MW|s1

- Acknowledgements: earlier: Right, I have it. / Okay, that's saved on my side. / Received, I have it. / Got it, I have the memo. / Okay, that's in. | latest: Thanks so much for helping me today.
- Coverage 0.7 (7 of 10); failed: R5 S-dep, R6 S-back, R9 S-span; hand-read replacements: 1; thinking tokens 586; words 309

```text
**Memo to Idris Mwangi**  
**Subject: Ashgrove Annual Partner Conference Update**  

**Summary**  
The Ashgrove Annual Partner Conference is now scheduled for **Wednesday 14 April 2027**, following a revised plan. Key workstreams, budget adjustments, and risks are outlined. Risks include delays from the keynote speaker’s visa status, late venue licensing, and potential sponsor withdrawals. The project budget has been reduced, and delegate numbers have increased.  

**Key facts**  
1. **Current conference opening day**: Wednesday 14 April 2027  
2. **Project budget after changes**: £283,000 (£301,000 – £18,000 reduction)  
3. **Speaker programme workstream owner**: Anders Whitfield  
4. **Risks delaying the opening day**: Keynote speaker visa delay, late venue licensing  
5. **Earliest ticket sales date**: Tuesday 10 February 2027 (8 working days after venue contract ends on 3 February 2027)  
6. **Latest registration sign-off date for catering kickoff**: Friday 8 March 2027 (8 working days before catering kickoff on 16 March 2027)  
7. **Audio-visual production workstream owner**: Lucía Castellanos  
8. **Catering tasting date**: Wednesday 17 November 2027  
9. **Working days between sponsorship checkpoint and venue booking review**: 8 working days (15 September 2027 to 23 September 2027)  
10. **Audio-visual production contract value**: £28,500  

**Next steps**  
1. Monitor the keynote speaker’s visa status and ensure venue licensing is finalized to avoid delays to the conference opening.  
2. Confirm the registration sign-off is completed by 8 March 2027 to enable the catering kickoff on 16 March 2027.  
3. Track catering costs and early registration numbers to manage the budget, as low registrations could impact financials.  
4. Ensure the speaker programme (Anders Whitfield) and audio-visual production (Lucía Castellanos) align with the revised conference timeline.  
5. Review the sponsorship checkpoint (15 September 2027) and venue booking review (23 September 2027) to confirm dependencies are met.  
6. Address the pending request for 16 additional delegate places and monitor sponsor withdrawal risks affecting delegate numbers.  

*Word count: 230*
```


## MM

### pilot_v1|h13|MM|s1

- Acknowledgements: earlier: Okay, that's come through. / Okay. / Okay, received. / Received, I have it. / All right, I've got that. | latest: I have it now.
- Coverage 0.6 (6 of 10); failed: R1 S-sum, R7 S-dep, R9 S-back, R10 S-span; hand-read replacements: 0; thinking tokens 3985; words 309

```text
**Memo to Anders Castellanos**  

**Summary**  
The Lumen Ridge CRM migration project is progressing with revised go-live dates, key milestones, and budget adjustments. Critical updates include risks to the go-live timeline, staffing dependencies, and financial considerations. The project now has a go-live date of **Monday 21 June 2027**, with ongoing monitoring of vendor timelines, data quality, and user adoption.  

**Key facts**  
1. **Risks delaying the go-live of the new CRM**:  
   - Incomplete customer data mappings  
   - Delays in single sign-on setup  
   - Late vendor fixes  
   - Low dashboard adoption  
   - Potential delays from the vendor demo  

2. **User training workstream owner**: **Elif Kaur**  

3. **Project budget after all approved changes**: **$186,000** (original $131,000 + $55,000 increase)  

4. **Date of the security review**: **Thursday 18 February 2027**  

5. **Data cleansing workstream owner**: **Rohan Brennan**  

6. **Current date of the go-live of the new CRM**: **Monday 21 June 2027**  

7. **Earliest date user acceptance testing can begin**: **Tuesday 14 February 2027** (5 working days after data migration ends on 9 February 2027)  

8. **Value of the vendor implementation contract**: **$70,000**  

9. **Latest date the vendor management sign-off can be completed for the reporting dashboards kickoff to go ahead on its planned date (21 October 2027)**: **Friday 20 October 2027** (3 working days before the kickoff date)  

10. **Working days between the access control checkpoint (15 September 2027) and the integration testing review (13 October 2027)**: **21 working days**  

**Next steps**  
- Monitor the impact of incomplete data mappings and single sign-on delays on the go-live date.  
- Track vendor sign-offs and ensure the reporting dashboards kickoff aligns with the planned date.  
- Address the pending request for 38 additional user licences to meet staffing needs.  
- Verify the status of the vendor demo and its potential impact on the go-live timeline.  
- Confirm the biosafety permit status, as delays could affect the first production sequencing run.  

[Word count: 290]
```

### pilot_v1|h4|MM|s0

- Acknowledgements: earlier: All right, I have the memo now. / Okay, I've saved that. / Okay, that has arrived. / Received, I have it. / Okay, that's saved on my side. | latest: Noted, I have it now.
- Coverage 0.6 (6 of 10); failed: R1 S-span, R5 S-sum, R8 S-back, R10 S-dep; hand-read replacements: 0; thinking tokens 3521; words 262

```text
**Summary**  
This memo outlines critical updates for the Lumen Ridge office relocation, including budget adjustments, key milestones, and workstream ownership. Key risks and dependencies are highlighted to ensure alignment with the project timeline and objectives.  

**Key facts**  
1. **Working days between lease negotiation checkpoint and network cabling review**: **23 working days** (from 22 September 2027 to 14 October 2027).  
2. **Value of the furniture supplier contract**: **£20,500**.  
3. **Project budget after changes**: Original budget of £91,000 increased by £49,000, totaling **£140,000**.  
4. **Date of the budget review**: **Thursday, 24 June 2027**.  
5. **Risks delaying the official opening**: Late server rack delivery, fire-safety certification delay, parking shortage, and rising furniture costs.  
6. **Current official opening date**: **Thursday, 17 June 2027**.  
7. **Furniture procurement workstream owner**: **Elif Demir**.  
8. **Latest date for security badges sign-off**: **Wednesday, 3 July 2027** (to enable staff communications kickoff on 7 July 2027).  
9. **Moving logistics workstream owner**: **Freya Marchetti**.  
10. **Earliest date staff move-in can begin**: **Monday, 26 February 2027** (8 working days after network cabling ends on 18 February 2027).  

**Next steps**  
Monitor risks such as delayed server racks, fire-safety certification, and parking shortages to avoid delays to the official opening. Ensure all workstreams align with the updated budget and deadlines, including the security badges sign-off by 3 July 2027. Verify the budget review on 24 June 2027 and confirm the staff move-in readiness by 26 February 2027. Track the lease negotiation checkpoint (22 September 2027) and network cabling review (14 October 2027) to ensure compliance with the revised timeline.  

(Word count: 230)
```

### pilot_v1|h2|MM|s2

- Acknowledgements: earlier: Right, I have it. / Okay, I've put it with the others. / All right, I have the memo now. / Okay. / All right, I've got that. | latest: Okay, that's saved on my side.
- Coverage 0.8 (8 of 10); failed: R3 S-span, R9 S-sum; hand-read replacements: 0; thinking tokens 2829; words 296

```text
**Memo**  
**To:** Svetlana Quiroga, Programme Director  
**From:** [Your Name]  
**Subject:** Kestrel Warehouse Automation Briefing  

**Summary**  
The Kestrel warehouse automation project is progressing under the revised plan, with the switch to automated picking scheduled for **Thursday 14 October 2027**. Key milestones, budget adjustments, and risks require close monitoring to ensure timely delivery.  

**Key facts**  
1. **Project budget after all approved changes**: **€215,000** (original €169,000 + €46,000 increase).  
2. **Earliest date robot installation can begin**: **Thursday 25 March 2027** (5 working days after floor reinforcement ends on 18 March 2027).  
3. **Working days between robot procurement checkpoint (25 June 2027) and safety training review (23 July 2027)**: **25 working days** (5 weeks).  
4. **Date of the operator training course**: **Monday 30 August 2027**.  
5. **Value of the maintenance contract**: **€27,000**.  
6. **Latest date for software integration sign-off**: **Monday 22 February 2027** (4 working days before the planned maintenance contracts kickoff on 26 February 2027).  
7. **Owner of the shift planning workstream**: **Corentin Okafor**.  
8. **Owner of the floor reinforcement workstream**: **Oluwaseun Esposito**.  
9. **Risks delaying the switch to automated picking**:  
   - Delayed customs clearance for robots.  
   - Failed load test on the mezzanine.  
   - Higher electricity costs.  
10. **Current date of the switch to automated picking**: **Thursday 14 October 2027**.  

**Next steps**  
The switch to automated picking on 14 October 2027 depends on timely completion of floor reinforcement, robot installation, and safety training. Workstreams must align to address risks, including customs delays, failed load tests, and rising costs. The software integration sign-off must be completed by 22 February 2027 to enable the maintenance contracts kickoff. The safety training review (23 July 2027) and operator training course (30 August 2027) are critical milestones. Further updates will be shared as they become available.  

[Your Name]  
[Your Position]  

(Word count: 245)
```

### pilot_v1|h1|MM|s0

- Acknowledgements: earlier: Okay, I have the file. / All right, I have the memo now. / Right, I have it. / Okay, that's in. / Okay, I've got the memo. | latest: All right, I've got that.
- Coverage 0.6 (6 of 10); failed: R1 S-dep, R2 S-back, R3 S-span, R9 S-sum; hand-read replacements: 1; thinking tokens 1350; words 280

```text
**Memo to: Oluwaseun Szabo**  
**Subject: Marrowfield Flagship Store Launch Update**  

**Summary**  
The Marrowfield flagship store launch is progressing with key milestones, budget adjustments, and workstream ownership. Risks such as late delivery of the till system, delayed signage permits, and higher rent require close monitoring. The store opening is now scheduled for **Tuesday 3 August 2027**, following a revised plan.  

**Key facts**  
1. **Earliest stock loading date**: **Friday 27 February 2027** (4 working days after shop fit-out ends on 23 February 2027).  
2. **Latest marketing campaign sign-off date**: **Wednesday 25 August 2027** (8 working days before the shop fit-out kickoff on 1 September 2027).  
3. **Working days between staff recruitment checkpoint (25 May 2027) and stock buying review (2 June 2027)**: **7 working days**.  
4. **Fire-safety inspection date**: **Monday 18 October 2027**.  
5. **Security workstream owner**: **Teodor Okafor**.  
6. **Project budget after changes**: **€437,000** (€443,000 original budget minus €6,000 March reduction).  
7. **Current store opening date**: **Tuesday 3 August 2027**.  
8. **Point-of-sale systems workstream owner**: **Corentin Oyelaran**.  
9. **Risks delaying the store opening**:  
   - Late delivery of the till system.  
   - Delay in the signage permit.  
   - Higher rent from the service charge.  
   - Weaker footfall forecasts affecting staff hires.  
10. **Shop-fitting contract value**: **€43,000**.  

**Next steps**  
- Monitor the till system delivery to avoid delays.  
- Ensure the signage permit is secured before the store opening.  
- Track rent costs to manage the budget.  
- Confirm staff recruitment readiness by the 25 May 2027 checkpoint.  
- Prepare for the fire-safety inspection on 18 October 2027.  
- Coordinate with Marisol Quiroga to finalize the marketing campaign sign-off by 25 August 2027.  
- Review stock loading and buying review timelines to align with the August opening.  

*Word count: 220*
```

### pilot_v1|h8|MM|s0

- Acknowledgements: earlier: Okay, I have it here. / All right, I've got that. / Okay, I've got the memo. / Okay, that's come through. / Noted, I have it now. | latest: Received, I have it.
- Coverage 0.6 (6 of 10); failed: R4 S-span, R5 S-back, R9 S-dep, R10 S-sum; hand-read replacements: 0; thinking tokens 1510; words 327

```text
**Summary**  
This briefing outlines the Kestrel classroom technology refresh project, including workstream ownership, budget adjustments, key milestones, and risks. The revised first lessons date is Monday 6 September 2027, with potential delays from content filter approvals, charging trolleys, and device breakage. The project budget has increased to £311,000 after a £34,000 approval, while a proposed £36,000 increase was rejected.  

**Key facts**  
1. **Joaquin Mwangi** owns the device procurement workstream.  
2. **Hamid Kaur** owns the safeguarding filters workstream.  
3. The current date of the first lessons on the new devices is **Monday, 6 September 2027** (revised plan).  
4. The network upgrade review (25 March 2027) takes place **19 working days** after the parent communications checkpoint (1 March 2027).  
5. The latest date the teacher training sign-off can be completed for the asset tagging kickoff (planned for 27 September 2027) is **Monday, 20 September 2027** (7 working days prior).  
6. The device leasing agreement is valued at **£29,500**.  
7. The teacher training day is scheduled for **Wednesday, 17 February 2027**.  
8. The project budget after all approved changes is **£311,000** (£277,000 original + £34,000 increase).  
9. The earliest date device deployment can begin is **Tuesday, 22 May 2027** (3 working days after the network upgrade ends on 19 May 2027).  
10. Risks that could delay the first lessons include late approval of the content filter, shortage of charging trolleys, and device breakage rates.  

**Next steps**  
Monitor progress on content filter approvals, charging trolleys, and device breakage to avoid delays to the first lessons on 6 September 2027. Ensure the teacher training sign-off is completed by 20 September 2027 to align with the asset tagging kickoff. Track the network upgrade (ends 19 May 2027) and parent communications checkpoint (1 March 2027) to confirm sequencing. Verify the revised first lessons date against all workstreams and address any risks from rising support fees or unapproved laptop requests. Confirm coordination between workstream owners to maintain project timelines.
```


## N

### pilot_v1|h11|N|s2

- Acknowledgements: none (cell N)
- Coverage 0.6 (6 of 10); failed: R3 S-span, R7 S-back, R8 S-sum, R10 S-dep; hand-read replacements: 0; thinking tokens 559; words 255

```text
**Memo to Priya Petrov**  

**Summary**  
This briefing outlines the revised plan for the store opening on Tuesday, 31 August 2027, including budget updates, workstream ownership, key milestones, and risks.  

**Key facts**  
1. **Marketing review** is scheduled for **Wednesday, 1 September 2027**.  
2. The **security workstream** is owned by **Marisol Kaur**.  
3. The **staff recruitment review** (18 August 2027) takes place **53 working days** after the marketing campaign checkpoint (23 July 2027).  
4. The **current store opening date** is **Tuesday, 31 August 2027**.  
5. The **shop fit-out workstream** is owned by **Oluwaseun Esposito**.  
6. The **project budget after all approved changes** is **€233,000** (€189,000 + €44,000 increase).  
7. The **latest date** the stock buying sign-off can be completed for the point-of-sale systems kickoff (planned for 28 September 2027) is **Friday, 23 September 2027** (5 working days prior).  
8. **Risks** that could delay the store opening include:  
   - Shortfall in seasonal stock.  
   - Delay in the signage permit.  
   - Security-related issues.  
   - Weaker footfall forecasts affecting the budget.  
9. The **shop-fitting contract** is valued at **€50,000**.  
10. The **earliest date stock loading can begin** is **Tuesday, 20 June 2027** (3 working days after the shop fit-out ends on 17 June 2027).  

**Next steps**  
Monitor key milestones: marketing review (1 September), staff recruitment review (18 August), and the point-of-sale systems kickoff (28 September). Address risks by ensuring timely sign-offs, securing signage permits, and managing seasonal stock procurement. Track staffing approvals and budget impacts from weaker footfall forecasts. Update the briefing with further workstream details as provided.  

[Word count: 280]
```

### pilot_v1|h19|N|s0

- Acknowledgements: none (cell N)
- Coverage 0.7 (7 of 10); failed: R4 S-back, R5 S-sum, R10 S-dep; hand-read replacements: 0; thinking tokens 3367; words 284

```text
**Memo to: Lucía Takahashi**  
**Subject: Tidewell Flagship Store Launch Briefing**  

**Summary**  
The Tidewell flagship store launch is now based on the revised plan of Tuesday 26 January 2027, with the store opening scheduled for Thursday 19 August 2027. Key updates include a €18,000 budget reduction, staffing adjustments, and critical dependencies on the shop fit-out and stock procurement. Risks such as seasonal stock shortfalls, delayed till systems, and service charge increases could impact the timeline.  

**Key facts**  
1. **Date of the stock count**: Wednesday, 18 August 2027.  
2. **Current store opening date**: Thursday, 19 August 2027.  
3. **Staff recruitment workstream owner**: Oluwaseun Nakamura.  
4. **Latest date for shop fit-out sign-off**: Monday, 22 July 2027 (to allow 8 working days for the point-of-sale systems kickoff on 30 July 2027).  
5. **Risks delaying the store opening**: Shortfall in seasonal stock, late till system delivery, dispute over lift installation, and higher service charge rent.  
6. **Value of the shop-fitting contract**: €25,500.  
7. **Working days between security checkpoint (12 July) and marketing campaign review (9 August)**: 20 working days.  
8. **Project budget after changes**: €556,000 (original €574,000 reduced by €18,000).  
9. **Stock buying workstream owner**: Amara Brennan.  
10. **Earliest date stock loading can begin**: Monday, 10 July 2027 (8 working days after shop fit-out ends on 3 June 2027).  

**Next steps**  
Monitor seasonal stock procurement, till system delivery, and service charge impacts to avoid delays. Ensure the shop fit-out sign-off is completed by 22 July 2027 to align with the planned point-of-sale systems kickoff. Track staffing and budget changes, and escalate risks to the steering group. Maintain communication with all workstreams for updates ahead of the August milestones.  

*End of memo*  

(Word count: 230)
```

### pilot_v1|h9|N|s1

- Acknowledgements: none (cell N)
- Coverage 0.7 (7 of 10); failed: R5 S-sum, R7 S-span, R9 S-back; hand-read replacements: 1; thinking tokens 730; words 346

```text
**Memo**  
**To:** Kwame Achterberg, Programme Director  
**From:** [Your Name]  
**Subject:** Veridian Eastside Clinic Opening Briefing  

**Summary**  
The Veridian Eastside clinic opening is governed by the revised plan of Monday 25 January 2027, which updates the first day of patient appointments to **Wednesday 3 November 2027**. Key milestones, budget adjustments, and risks are outlined, including potential delays from unfilled nurse vacancies, late X-ray unit delivery, and licensing backlogs.  

**Key facts**  
1. The **project budget after all approved changes** is **€426,000** (€399,000 original + €27,000 increase).  
2. The **clinical staffing workstream** is owned by **Zainab Iyer**.  
3. The **governance board meeting** is scheduled for **Wednesday, 4 August 2027**.  
4. The **current date of the first day of patient appointments** is **Wednesday, 3 November 2027**.  
5. Risks that could delay the first day of patient appointments include **unfilled nurse vacancies**, **late delivery of the X-ray unit**, and **backlogs at the licensing authority**.  
6. The **building fit-out workstream** is owned by **Svetlana Lindqvist**.  
7. The **regulatory approval review** (Tuesday, 4 May 2027) takes place **7 working days** after the **community outreach checkpoint** (Monday, 26 April 2027).  
8. The **value of the imaging equipment lease** is **€59,000**.  
9. The **latest date** the equipment procurement sign-off can be completed for the patient records kickoff (Friday, 17 September 2027) is **Friday, 10 August 2027**.  
10. The **earliest date equipment installation can begin** is **Thursday, 10 June 2027**, three working days after the building fit-out ends on Monday, 7 June 2027.  

**Next steps**  
- Monitor the revised timeline for the first day of patient appointments and address risks such as unfilled nurse vacancies and X-ray unit delays.  
- Ensure the equipment procurement sign-off is completed by 10 August 2027 to align with the patient records kickoff.  
- Track the regulatory approval review (4 May 2027) and community outreach checkpoint (26 April 2027) to confirm compliance.  
- Verify the governance board meeting (4 August 2027) and staff induction day (12 July 2027) are aligned with other milestones.  
- Address potential budget impacts from higher energy tariffs and ensure the imaging equipment lease is secured.  

[Word count: 258]
```

### pilot_v1|h0|N|s1

- Acknowledgements: none (cell N)
- Coverage 0.7 (7 of 10); failed: R2 S-span, R6 S-back, R8 S-dep; hand-read replacements: 0; thinking tokens 758; words 338

```text
**Memo to Aisling Okafor**  
**Subject: Kestrel Flagship Store Launch Briefing**  

**Summary**  
This memo outlines critical updates and risks for the Kestrel flagship store launch, including revised timelines, budget adjustments, and key workstreams. The store opening is now scheduled for **Friday 25 June 2027**, following a revised plan from **Thursday 21 January 2027**. The project budget has been reduced by €22,000 to **€227,000**, and risks such as seasonal stock shortfalls and late till system delivery could delay the opening. Key workstreams and milestones are detailed below.  

**Key facts**  
1. **Risks delaying the store opening**: Seasonal stock shortfall and late delivery of the till system.  
2. **Working days between stock buying checkpoint (6 September) and security review (27 September)**: 21 working days.  
3. **Project budget after changes**: €227,000 (original €249,000 reduced by €22,000).  
4. **Shop fit-out workstream owner**: Priya Marchetti.  
5. **Fire-safety inspection date**: Wednesday, 22 September 2027.  
6. **Latest date for marketing campaign sign-off**: Monday, 17 February 2027 (to allow 5 working days for the point-of-sale systems kickoff on 22 February 2027).  
7. **Staff recruitment workstream owner**: Marisol Quiroga.  
8. **Earliest stock loading date**: Friday, 29 May 2027 (3 working days after the shop fit-out ends on 26 May 2027).  
9. **Current store opening date**: Friday, 25 June 2027.  
10. **Shop-fitting contract value**: €40,500.  

**Next steps**  
- Monitor seasonal stock availability and till system delivery to mitigate opening delays.  
- Track the stock buying checkpoint (6 September) and ensure the security review (27 September) aligns with the 21-working-day gap.  
- Confirm the marketing campaign sign-off is completed by 17 February 2027 to enable the point-of-sale systems kickoff on 22 February 2027.  
- Verify staffing updates, including the pending approval for 2 additional hires, and assess the impact of higher service charges on recruitment.  
- Monitor the fire-safety inspection (22 September 2027) and press preview (9 November 2027) as critical pre-opening milestones.  
- Ensure Priya Marchetti’s shop fit-out workstream adheres to the 26 May 2027 completion date to allow stock loading to start on 29 May 2027.  

*Word count: 260*
```

### pilot_v1|h0|N|s0

- Acknowledgements: none (cell N)
- Coverage 0.7 (7 of 10); failed: R1 S-sum, R6 S-back, R8 S-dep; hand-read replacements: 0; thinking tokens 2052; words 312

```text
**Memo to Aisling Okafor**  
**Subject: Kestrel Flagship Store Launch Briefing**  

**Summary**  
This memo outlines critical updates and risks for the Kestrel flagship store launch, including delays, deadlines, and workstream ownership. The store opening is now scheduled for Friday 25 June 2027, with key milestones and dependencies highlighted.  

**Key facts**  
1. **Risks delaying the store opening**: Seasonal stock shortfall, late delivery of the till system, and potential delays in the shop fit-out or security review.  
2. **Working days between stock buying checkpoint (6 September 2027) and security review (27 September 2027)**: 15 working days (Monday to Friday, no public holidays).  
3. **Project budget after changes**: €227,000 (original €249,000 reduced by €22,000 in March).  
4. **Shop fit-out workstream owner**: Priya Marchetti.  
5. **Fire-safety inspection date**: Wednesday, 22 September 2027.  
6. **Latest marketing campaign sign-off date for point-of-sale systems kickoff (planned for 22 February 2027)**: Monday, 17 February 2027 (5 working days prior).  
7. **Staff recruitment workstream owner**: Marisol Quiroga.  
8. **Earliest stock loading date**: Monday, 29 May 2027 (3 working days after shop fit-out ends on 26 May 2027).  
9. **Current store opening date**: Friday, 25 June 2027.  
10. **Shop-fitting contract value**: €40,500.  

**Next steps**  
- Monitor the shop fit-out (ends 26 May 2027) and ensure the till system is delivered on time to avoid delays.  
- Track the stock buying checkpoint (6 September 2027) and security review (27 September 2027) to align with the fire-safety inspection (22 September 2027).  
- Confirm the marketing campaign sign-off by 17 February 2027 to enable the point-of-sale systems kickoff on 22 February 2027.  
- Verify staffing updates (29 hires + 4 approved, pending 2 more) and address weaker footfall forecasts impacting the budget.  
- Ensure Priya Marchetti and Marisol Quiroga report on stock loading (29 May 2027) and staff training (21 April 2027), respectively.  
- Monitor seasonal stock availability and the service charge’s impact on staffing numbers.  

*Word count: 240*
```
