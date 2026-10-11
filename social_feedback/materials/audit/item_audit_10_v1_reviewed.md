# Item audit: 10 calibration triples

For each item: read the fact sheet (ground truth, written by code), then the briefing (written by Sonnet from the fact sheet), then the requirements with the code-computed answers and the reference memo. Working days: Monday-Friday, no holidays; 'N working days after X' = the N-th weekday after X. Any defect means the generator gets fixed and 10 new triples are audited (DESIGN.md §4.2).


---

## cal-s0-L3 — CRM migration, level 3

### Fact sheet

- A proposed further increase of $21,000 was rejected by the steering group.
- A request for 26 more user licences is still awaiting approval.
- An additional 12 user licences were approved in March.
- An email of Friday 8 January 2027 gave the go-live of the new CRM as Friday 27 August 2027.
- Bastian Okafor owns the access control workstream.
- Freya Delacroix supports the data cleansing workstream, which Hamid Marchetti owns, and attends its meetings.
- Hamid Marchetti owns the data cleansing workstream.
- Ingrid Brennan owns the integration testing workstream.
- Marisol Haddad owns the user training workstream.
- Priya Mwangi owns the vendor management workstream.
- Risk: a delay in single sign-on setup could delay the go-live of the new CRM.
- Risk: incomplete customer data mappings could delay the go-live of the new CRM.
- Risk: licence price increases could affect the budget.
- Risk: low dashboard adoption could affect the number of user licences.
- Sanna Kaur supports the access control workstream, which Bastian Okafor owns, and attends its meetings.
- The approved project budget is $544,000.
- The data migration is scheduled to end on Friday 19 March 2027.
- The integration testing review is on Wednesday 26 May 2027.
- The plan provides for 385 user licences.
- The reporting dashboards checkpoint is on Tuesday 27 April 2027.
- The revised plan of Wednesday 27 January 2027 sets the go-live of the new CRM for Monday 22 November 2027.
- The revised plan of Wednesday 27 January 2027 supersedes all earlier emails.
- The steering committee meeting is on Wednesday 10 November 2027.
- The steering group approved an increase of $47,000 to the project budget in March.
- The trainer briefing is on Friday 30 April 2027.
- The user training kickoff can begin 3 working days after the vendor management sign-off is completed.
- The user training kickoff is planned for Thursday 7 October 2027.
- The vendor demo is on Friday 15 October 2027.
- The vendor implementation contract is worth $47,500.
- User acceptance testing can begin 10 working days after the data migration ends.
- Working days are Monday to Friday; there are no public holidays in this period.

### Briefing

The Lumen Ridge CRM migration is now planned around the revised plan of Wednesday 27 January 2027, which sets the go-live of the new CRM for Monday 22 November 2027. That plan supersedes all earlier emails, including the one of Friday 8 January 2027 that gave the go-live as Friday 27 August 2027. The steering committee meeting is on Wednesday 10 November 2027, and the vendor demo is on Friday 15 October 2027.

Workstream ownership is settled. Hamid Marchetti owns the data cleansing workstream, and Freya Delacroix supports it and attends its meetings. Bastian Okafor owns the access control workstream, with Sanna Kaur supporting it and attending its meetings. Marisol Haddad owns user training, Priya Mwangi owns vendor management, and Ingrid Brennan owns integration testing.

On schedule, the data migration is scheduled to end on Friday 19 March 2027. Working days are Monday to Friday, and there are no public holidays in this period. The reporting dashboards checkpoint is on Tuesday 27 April 2027, followed by the trainer briefing on Friday 30 April 2027. The integration testing review is on Wednesday 26 May 2027.

User acceptance testing can begin 10 working days after the data migration ends. Likewise, the user training kickoff can begin 3 working days after the vendor management sign-off is completed, and it is planned for Thursday 7 October 2027.

On budget, the approved project budget is $544,000. The steering group approved an increase of $47,000 to the project budget in March, while a proposed further increase of $21,000 was rejected by the steering group. The vendor implementation contract is worth $47,500.

Licensing follows a similar pattern. The plan provides for 385 user licences, and an additional 12 user licences were approved in March. A request for 26 more user licences is still awaiting approval.

Several risks are being tracked. A delay in single sign-on setup could delay the go-live of the new CRM, and incomplete customer data mappings could do the same. Low dashboard adoption could affect the number of user licences, and licence price increases could affect the budget.

### Requirements and correct answers

R1. how many working days after the reporting dashboards checkpoint the integration testing review takes place — **21**
R2. the earliest date user acceptance testing can begin — **Friday 2 April 2027**
R3. the project budget after all approved changes — **$591,000**
R4. the risks that could delay the go-live of the new CRM — **exactly these risks: incomplete customer data mappings; a delay in single sign-on setup. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R5. who owns the data cleansing workstream — **Hamid Marchetti**
R6. the latest date the vendor management sign-off can be completed for the user training kickoff to go ahead on its planned date — **Monday 4 October 2027**
R7. the current date of the go-live of the new CRM — **Monday 22 November 2027**
R8. who owns the access control workstream — **Bastian Okafor**
R9. the value of the vendor implementation contract — **$47,500**
R10. the date of the steering committee meeting — **Wednesday 10 November 2027**

### Reference memo

> **Memo**
> 
> **To:** Amara Valdés, Programme Director
> **Re:** Lumen Ridge CRM migration status
> 
> **Summary**
> 
> The migration is planned around the revised plan of Wednesday 27 January 2027, which supersedes all earlier emails, including the one of 8 January. The go-live of the new CRM is currently Monday 22 November 2027. Two risks could delay it, and the other points below are settled.
> 
> **Key facts**
> 
> - **Go-live:** Monday 22 November 2027.
> - **Go-live risks:** a delay in single sign-on setup, and incomplete customer data mappings.
> - **Steering committee meeting:** Wednesday 10 November 2027.
> - **Integration testing review:** Wednesday 26 May 2027, 21 working days after the reporting dashboards checkpoint on Tuesday 27 April 2027.
> - **User acceptance testing:** the data migration ends on Friday 19 March 2027, so testing can begin no earlier than Friday 2 April 2027.
> - **Budget:** $591,000 after all approved changes (the approved $544,000 plus the $47,000 increase approved in March).
> - **Vendor implementation contract:** $47,500.
> - **Data cleansing workstream:** owned by Hamid Marchetti.
> - **Access control workstream:** owned by Bastian Okafor.
> - **Training kickoff:** planned for Thursday 7 October 2027. The vendor management sign-off must therefore be completed by Monday 4 October 2027 at the latest.
> 
> **Next steps**
> 
> 1. Track the single sign-on setup and the customer data mappings closely, as they are the two risks to the 22 November go-live.
> 2. Ask Priya Mwangi to confirm that the vendor management sign-off will be completed by Monday 4 October 2027.
> 3. Prepare for the steering committee meeting on 10 November 2027.
> 4. Schedule user acceptance testing to start from Friday 2 April 2027.

**Checks** (tick, or write the problem underneath):

- [x] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [x] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

PASS. Checked owner/supporter distinctions, source precedence, budget exclusions and the two go-live risks. Recalculated the three working-day answers: 21 days; earliest UAT 2 April; latest sign-off 4 October. All ten reference answers agree with the fact sheet.


---

## cal-s1-L3 — store launch, level 3

### Fact sheet

- A proposed further increase of €35,000 was rejected by the steering group.
- A request for 3 more staff hires is still awaiting approval.
- An additional 3 staff hires were approved in March.
- An email of Thursday 7 January 2027 gave the store opening as Monday 23 August 2027.
- Corentin Iyer owns the shop fit-out workstream.
- Emeka Takahashi owns the stock buying workstream.
- Freya Kaur supports the stock buying workstream, which Emeka Takahashi owns, and attends its meetings.
- Marisol Halloran owns the marketing campaign workstream.
- Rafael Marchetti supports the shop fit-out workstream, which Corentin Iyer owns, and attends its meetings.
- Risk: a delay in the signage permit could delay the store opening.
- Risk: higher rent from the service charge could affect the number of staff hires.
- Risk: late delivery of the till system could delay the store opening.
- Risk: rising wage costs could affect the budget.
- Stock loading can begin 10 working days after the shop fit-out ends.
- Teodor Bakshi owns the point-of-sale systems workstream.
- The approved project budget is €301,000.
- The fire-safety inspection is on Friday 30 April 2027.
- The marketing campaign review is on Friday 3 September 2027.
- The plan provides for 36 staff hires.
- The press preview is on Friday 26 November 2027.
- The revised plan of Wednesday 27 January 2027 sets the store opening for Wednesday 7 July 2027.
- The revised plan of Wednesday 27 January 2027 supersedes all earlier emails.
- The security kickoff can begin 3 working days after the point-of-sale systems sign-off is completed.
- The security kickoff is planned for Tuesday 25 May 2027.
- The shop fit-out is scheduled to end on Tuesday 8 June 2027.
- The shop-fitting contract is worth €51,500.
- The staff recruitment checkpoint is on Monday 2 August 2027.
- The staff training day is on Tuesday 26 October 2027.
- The steering group approved an increase of €31,000 to the project budget in March.
- Working days are Monday to Friday; there are no public holidays in this period.
- Yuki Delacroix owns the security workstream.

### Briefing

The Orrin flagship store launch is now working to the revised plan of Wednesday 27 January 2027, which sets the store opening for Wednesday 7 July 2027 and supersedes all earlier emails. One of those earlier emails, dated Thursday 7 January 2027, had given the opening as Monday 23 August 2027. Working days are Monday to Friday, and there are no public holidays in this period.

On ownership, Corentin Iyer owns the shop fit-out workstream, with Rafael Marchetti supporting it and attending its meetings. The shop-fitting contract is worth €51,500. The shop fit-out is scheduled to end on Tuesday 8 June 2027, and the fire-safety inspection is on Friday 30 April 2027. Teodor Bakshi owns the point-of-sale systems workstream, and the security kickoff can begin 3 working days after the point-of-sale systems sign-off is completed. The security kickoff itself is planned for Tuesday 25 May 2027, and Yuki Delacroix owns the security workstream.

Emeka Takahashi owns the stock buying workstream, and Freya Kaur supports it and attends its meetings. Stock loading can begin 10 working days after the shop fit-out ends. Marisol Halloran owns the marketing campaign workstream, whose review is on Friday 3 September 2027. The press preview is on Friday 26 November 2027, and the staff training day is on Tuesday 26 October 2027.

On budget, the approved project budget is €301,000. The steering group approved an increase of €31,000 to the project budget in March, while a proposed further increase of €35,000 was rejected by the steering group. Rising wage costs are flagged as a risk that could affect the budget.

On staffing, the plan provides for 36 staff hires, and an additional 3 staff hires were approved in March. A request for 3 more staff hires is still awaiting approval. The staff recruitment checkpoint is on Monday 2 August 2027. Higher rent from the service charge is recorded as a risk that could affect the number of staff hires.

Two further risks concern the opening date directly. Late delivery of the till system could delay the store opening, and a delay in the signage permit could likewise delay the store opening. Both sit alongside the budget and staffing risks above as the items the team is tracking against the revised plan.

### Requirements and correct answers

R1. the project budget after all approved changes — **€332,000**
R2. how many working days after the staff recruitment checkpoint the marketing campaign review takes place — **24**
R3. the current date of the store opening — **Wednesday 7 July 2027**
R4. who owns the shop fit-out workstream — **Corentin Iyer**
R5. the latest date the point-of-sale systems sign-off can be completed for the security kickoff to go ahead on its planned date — **Thursday 20 May 2027**
R6. the date of the staff training day — **Tuesday 26 October 2027**
R7. who owns the stock buying workstream — **Emeka Takahashi**
R8. the value of the shop-fitting contract — **€51,500**
R9. the risks that could delay the store opening — **exactly these risks: a delay in the signage permit; late delivery of the till system. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R10. the earliest date stock loading can begin — **Tuesday 22 June 2027**

### Reference memo

> **Memo to Noor Whitfield, Programme Director**
> 
> **Summary**
> 
> The Orrin flagship store launch is working to the revised plan of Wednesday 27 January 2027. This plan supersedes all earlier emails, including the one of 7 January that gave a different opening date. The store opening is now Wednesday 7 July 2027. Two risks could delay the opening: a delay in the signage permit and late delivery of the till system.
> 
> **Key facts**
> 
> - **Project budget:** €332,000 after all approved changes. This is the original €301,000 plus the €31,000 increase approved in March. The further €35,000 proposed was rejected.
> - **Store opening:** Wednesday 7 July 2027.
> - **Shop fit-out owner:** Corentin Iyer.
> - **Shop-fitting contract:** €51,500.
> - **Stock buying owner:** Emeka Takahashi.
> - **Stock loading:** the earliest start is Tuesday 22 June 2027, which is 10 working days after the shop fit-out ends on 8 June.
> - **Security kickoff:** planned for Tuesday 25 May 2027, so the point-of-sale systems sign-off must be completed by Thursday 20 May 2027 at the latest.
> - **Marketing campaign review:** Friday 3 September 2027, which is 24 working days after the staff recruitment checkpoint on Monday 2 August 2027.
> - **Staff training day:** Tuesday 26 October 2027.
> - **Risks that could delay the opening:** a delay in the signage permit and late delivery of the till system.
> 
> **Next steps**
> 
> - Track the signage permit and the till system delivery closely, as they are the two items that could move the opening date.
> - Confirm that Teodor Bakshi's point-of-sale sign-off will be completed by Thursday 20 May 2027.
> - Keep all workstream owners working to the revised plan of 27 January 2027 rather than to earlier emails.

**Checks** (tick, or write the problem underneath):

- [x] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [x] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

PASS. Recalculated 24 working days, earliest loading 22 June and latest sign-off 20 May. Budget is €301,000 + €31,000 = €332,000. Owners, contract, training date, revised opening date and exact risk set agree.


---

## cal-s2-L3 — school IT refresh, level 3

### Fact sheet

- A proposed further increase of £38,000 was rejected by the steering group.
- A request for 3 more laptops is still awaiting approval.
- An additional 15 laptops were approved in March.
- An email of Friday 15 January 2027 gave the first lessons on the new devices as Tuesday 3 August 2027.
- Bastian Szabo supports the asset tagging workstream, which Dmitri Kaur owns, and attends its meetings.
- Device deployment can begin 5 working days after the network upgrade ends.
- Dmitri Kaur owns the asset tagging workstream.
- Emeka Quiroga owns the safeguarding filters workstream.
- Noor Kowalczyk owns the parent communications workstream.
- Risk: a shortage of charging trolleys could delay the first lessons on the new devices.
- Risk: device breakage rates could affect the number of laptops.
- Risk: higher insurance costs could affect the budget.
- Risk: late approval of the content filter could delay the first lessons on the new devices.
- Thandiwe Fontaine owns the device procurement workstream.
- The approved project budget is £563,000.
- The device leasing agreement is worth £53,000.
- The device procurement review is on Monday 16 August 2027.
- The governors' meeting is on Monday 5 April 2027.
- The network upgrade is scheduled to end on Wednesday 14 April 2027.
- The parent communications kickoff can begin 6 working days after the network upgrade sign-off is completed.
- The parent communications kickoff is planned for Wednesday 4 August 2027.
- The parents' evening is on Thursday 11 November 2027.
- The plan provides for 388 laptops.
- The revised plan of Thursday 21 January 2027 sets the first lessons on the new devices for Wednesday 12 May 2027.
- The revised plan of Thursday 21 January 2027 supersedes all earlier emails.
- The safeguarding filters checkpoint is on Tuesday 13 July 2027.
- The steering group approved a reduction of £48,000 to the project budget in March.
- The supplier demo is on Wednesday 3 March 2027.
- Working days are Monday to Friday; there are no public holidays in this period.
- Yuki Whitfield supports the teacher training workstream, which Zainab Abernethy owns, and attends its meetings.
- Zainab Abernethy owns the teacher training workstream.

### Briefing

The Ashgrove classroom technology refresh is working to a revised plan dated Thursday 21 January 2027, which supersedes all earlier emails. That plan sets the first lessons on the new devices for Wednesday 12 May 2027. An earlier email of Friday 15 January 2027 had given Tuesday 3 August 2027 for the same milestone, but it is covered by the supersession. The network upgrade is scheduled to end on Wednesday 14 April 2027, and the plan provides for 388 laptops.

On the money side, the approved project budget is £563,000. In March the steering group approved a reduction of £48,000 to the project budget, and an additional 15 laptops were approved in the same month. A request for 3 more laptops is still awaiting approval. A proposed further increase of £38,000 was rejected by the steering group. Separately, the device leasing agreement is worth £53,000.

Ownership is spread across five workstreams. Thandiwe Fontaine owns device procurement, and Emeka Quiroga owns safeguarding filters. Dmitri Kaur owns asset tagging, with Bastian Szabo supporting that workstream and attending its meetings. Zainab Abernethy owns teacher training, supported by Yuki Whitfield, who also attends its meetings. Noor Kowalczyk owns parent communications.

Several timing rules and dates sit around the network work. Device deployment can begin 5 working days after the network upgrade ends. The parent communications kickoff can begin 6 working days after the network upgrade sign-off is completed, and it is planned for Wednesday 4 August 2027. Working days are Monday to Friday, and there are no public holidays in this period.

The fixed calendar includes the supplier demo on Wednesday 3 March 2027 and the governors' meeting on Monday 5 April 2027. The safeguarding filters checkpoint is on Tuesday 13 July 2027, and the device procurement review is on Monday 16 August 2027. The parents' evening is on Thursday 11 November 2027.

Four risks are on the register. Higher insurance costs could affect the budget. Device breakage rates could affect the number of laptops. A shortage of charging trolleys could delay the first lessons on the new devices, and late approval of the content filter could do the same.

### Requirements and correct answers

R1. how many working days after the safeguarding filters checkpoint the device procurement review takes place — **24**
R2. who owns the teacher training workstream — **Zainab Abernethy**
R3. the project budget after all approved changes — **£515,000**
R4. the earliest date device deployment can begin — **Wednesday 21 April 2027**
R5. the risks that could delay the first lessons on the new devices — **exactly these risks: a shortage of charging trolleys; late approval of the content filter. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R6. who owns the asset tagging workstream — **Dmitri Kaur**
R7. the latest date the network upgrade sign-off can be completed for the parent communications kickoff to go ahead on its planned date — **Tuesday 27 July 2027**
R8. the current date of the first lessons on the new devices — **Wednesday 12 May 2027**
R9. the value of the device leasing agreement — **£53,000**
R10. the date of the supplier demo — **Wednesday 3 March 2027**

### Reference memo

> **Memo to Tomasz Nakamura, Programme Director**
> 
> **Summary**
> 
> The Ashgrove classroom technology refresh is following the revised plan of Thursday 21 January 2027, which supersedes all earlier emails. The first lessons on the new devices are now set for Wednesday 12 May 2027. Two risks could delay that date, and both are covered below.
> 
> **Key facts**
> 
> - **First lessons:** Wednesday 12 May 2027.
> - **Budget:** After all approved changes, the project budget is £515,000.
> - **Device deployment:** The earliest start is Wednesday 21 April 2027.
> - **Parent communications kickoff:** It is planned for Wednesday 4 August 2027. For it to go ahead on that date, network upgrade sign-off must be completed by Tuesday 27 July 2027 at the latest.
> - **Leasing:** The device leasing agreement is worth £53,000.
> - **Supplier demo:** Wednesday 3 March 2027.
> - **Procurement review:** It takes place 24 working days after the safeguarding filters checkpoint.
> - **Ownership:** Zainab Abernethy owns teacher training. Dmitri Kaur owns asset tagging.
> - **Risks to the first lessons:** A shortage of charging trolleys could delay them. Late approval of the content filter could also delay them.
> 
> **Next steps**
> 
> 1. Monitor the charging trolley supply and the content filter approval closely, since either could move the first lessons.
> 2. Keep network upgrade sign-off on track so that the parent communications kickoff holds its planned date.
> 3. Prepare device deployment so that it can begin as soon as the network upgrade allows.
> 4. Ask workstream owners to confirm readiness ahead of the governors' meeting on Monday 5 April 2027.
> 
> Please let me know if you would like further detail on any point.

**Checks** (tick, or write the problem underneath):

- [x] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [x] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

PASS. Recalculated 24 working days, earliest deployment 21 April and latest sign-off 27 July. Budget is £563,000 - £48,000 = £515,000. Ownership, supplier demo, revised lessons date and the two delay risks agree.


---

## cal-s3-L3 — office relocation, level 3

### Fact sheet

- A proposed further increase of £23,000 was rejected by the steering group.
- A request for 3 more desks is still awaiting approval.
- Amara Okafor owns the lease negotiation workstream.
- An additional 7 desks were approved in March.
- An email of Tuesday 5 January 2027 gave the official opening of the new office as Monday 11 October 2027.
- Bastian Esposito owns the network cabling workstream.
- Corentin Valdés supports the network cabling workstream, which Bastian Esposito owns, and attends its meetings.
- Elif Delacroix owns the security badges workstream.
- Hamid Fontaine supports the lease negotiation workstream, which Amara Okafor owns, and attends its meetings.
- Marisol Mensah owns the staff communications workstream.
- Risk: a delay in fire-safety certification could delay the official opening of the new office.
- Risk: a shortage of parking spaces could affect the budget.
- Risk: late delivery of the server racks could delay the official opening of the new office.
- Risk: rising removal costs could affect the number of desks.
- Thandiwe Brennan owns the furniture procurement workstream.
- The approved project budget is £496,000.
- The furniture delivery is on Thursday 25 March 2027.
- The furniture supplier contract is worth £65,000.
- The landlord walkthrough is on Wednesday 17 March 2027.
- The moving logistics kickoff can begin 7 working days after the furniture procurement sign-off is completed.
- The moving logistics kickoff is planned for Friday 25 June 2027.
- The network cabling installation is scheduled to end on Wednesday 9 June 2027.
- The plan provides for 133 desks.
- The revised plan of Wednesday 20 January 2027 sets the official opening of the new office for Thursday 4 November 2027.
- The revised plan of Wednesday 20 January 2027 supersedes all earlier emails.
- The security audit is on Thursday 25 February 2027.
- The security badges review is on Wednesday 13 October 2027.
- The staff communications checkpoint is on Thursday 16 September 2027.
- The staff move-in can begin 6 working days after the network cabling installation ends.
- The steering group approved a reduction of £29,000 to the project budget in March.
- Working days are Monday to Friday; there are no public holidays in this period.

### Briefing

The Kestrel office relocation is being run through several workstreams, each with a named owner. Amara Okafor owns the lease negotiation workstream, and Hamid Fontaine supports it and attends its meetings. Bastian Esposito owns the network cabling workstream, which Corentin Valdés supports and whose meetings he attends. Thandiwe Brennan owns furniture procurement, Marisol Mensah owns staff communications, and Elif Delacroix owns security badges.

On dates, the official opening of the new office was given as Monday 11 October 2027 in an email of Tuesday 5 January 2027. The revised plan of Wednesday 20 January 2027 supersedes all earlier emails, and it sets the opening for Thursday 4 November 2027. The landlord walkthrough is on Wednesday 17 March 2027, and the furniture delivery follows on Thursday 25 March 2027. The security audit is on Thursday 25 February 2027, so it comes before both of those.

The network cabling installation is scheduled to end on Wednesday 9 June 2027. The moving logistics kickoff is planned for Friday 25 June 2027, and it can begin 7 working days after the furniture procurement sign-off is completed. Working days are Monday to Friday, and there are no public holidays in this period.

On the financial side, the approved project budget is £496,000. The furniture supplier contract is worth £65,000. In March the steering group approved a reduction of £29,000 to the project budget, while a proposed further increase of £23,000 was rejected by the steering group. Desk numbers have also moved: the plan provides for 133 desks, an additional 7 desks were approved in March, and a request for 3 more desks is still awaiting approval.

Later milestones are the staff communications checkpoint on Thursday 16 September 2027 and the security badges review on Wednesday 13 October 2027. The staff move-in can begin 6 working days after the network cabling installation ends, which ties the occupation of the space directly to the cabling workstream's progress.

Three risks are currently recorded. Late delivery of the server racks could delay the official opening of the new office, and a delay in fire-safety certification could do the same. A shortage of parking spaces could affect the budget, and rising removal costs could affect the number of desks. These are noted alongside the schedule so that the workstream owners can see how the different items connect.

### Requirements and correct answers

R1. how many working days after the staff communications checkpoint the security badges review takes place — **19**
R2. the risks that could delay the official opening of the new office — **exactly these risks: late delivery of the server racks; a delay in fire-safety certification. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R3. the latest date the furniture procurement sign-off can be completed for the moving logistics kickoff to go ahead on its planned date — **Wednesday 16 June 2027**
R4. who owns the lease negotiation workstream — **Amara Okafor**
R5. the earliest date the staff move-in can begin — **Thursday 17 June 2027**
R6. the value of the furniture supplier contract — **£65,000**
R7. the current date of the official opening of the new office — **Thursday 4 November 2027**
R8. who owns the network cabling workstream — **Bastian Esposito**
R9. the project budget after all approved changes — **£467,000**
R10. the date of the security audit — **Thursday 25 February 2027**

### Reference memo

> **Memo to Tomasz Nakamura, Programme Director**
> **Subject: Kestrel office relocation status**
> 
> **Summary**
> 
> The Kestrel relocation is on track against the revised plan of 20 January 2027, which supersedes all earlier emails. The official opening is now Thursday 4 November 2027. Two recorded risks could delay it, and the budget has been reduced by an approved change.
> 
> **Key facts**
> 
> - **Official opening:** The current date is Thursday 4 November 2027. The earlier date of 11 October 2027 is superseded.
> - **Risks to the opening:** Late delivery of the server racks, and a delay in fire-safety certification, could each delay it.
> - **Lease negotiation owner:** Amara Okafor.
> - **Network cabling owner:** Bastian Esposito.
> - **Security audit:** Thursday 25 February 2027.
> - **Furniture supplier contract:** £65,000.
> - **Project budget:** £467,000 after all approved changes. This is the £496,000 approved budget less the £29,000 reduction approved in March. The proposed £23,000 increase was rejected.
> - **Staff move-in:** The earliest start is Thursday 17 June 2027, six working days after cabling ends on 9 June 2027.
> - **Badges review:** It takes place 19 working days after the staff communications checkpoint of 16 September 2027, on 13 October 2027.
> 
> **Next steps**
> 
> - Complete the furniture procurement sign-off by Wednesday 16 June 2027 at the latest. This keeps the moving logistics kickoff on its planned date of Friday 25 June 2027, which needs seven working days after sign-off.
> - Thandiwe Brennan should confirm the sign-off timing.
> - Bastian Esposito should keep the cabling end date of 9 June 2027, because the staff move-in depends on it.
> - Workstream owners should monitor the server rack delivery and fire-safety certification, since both could affect the opening date.

**Checks** (tick, or write the problem underneath):

- [ ] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [ ] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [ ] The reference memo is correct.

Notes:

FAIL — briefing and reference defects. The briefing says "Three risks are currently recorded" but the fact sheet and the following sentences identify FOUR distinct risks: server racks, fire-safety certification, parking spaces, and removal costs. Change Three to Four. The reference also adds "on track", which the source does not establish; replace that status claim with "is working to the revised plan". The ten requested answers themselves are correct: 19 working days; latest sign-off 16 June; earliest move-in 17 June; budget £467,000. Checks 1, 2 and 4 are left unticked for these defects.


---

## cal-s4-L3 — conference, level 3

### Fact sheet

- A proposed further increase of £54,000 was rejected by the steering group.
- A request for 30 more delegate places is still awaiting approval.
- An additional 80 delegate places were approved in March.
- An email of Thursday 14 January 2027 gave the conference opening day as Thursday 15 July 2027.
- Elif Sørensen owns the venue booking workstream.
- Hamid Valdés owns the sponsorship workstream.
- Kwame Bakshi owns the catering workstream.
- Lucía Esposito supports the catering workstream, which Kwame Bakshi owns, and attends its meetings.
- Noor Okafor owns the audio-visual production workstream.
- Risk: a sponsor withdrawing could affect the budget.
- Risk: late venue licensing could delay the conference opening day.
- Risk: low early registrations could affect the number of delegate places.
- Risk: the keynote speaker's visa delay could delay the conference opening day.
- Rohan Mensah supports the venue booking workstream, which Elif Sørensen owns, and attends its meetings.
- The approved project budget is £106,000.
- The audio-visual production contract is worth £42,000.
- The catering tasting is on Monday 19 April 2027.
- The dress rehearsal is on Thursday 12 August 2027.
- The plan provides for 526 delegate places.
- The registration review is on Thursday 8 April 2027.
- The revised plan of Thursday 28 January 2027 sets the conference opening day for Wednesday 14 July 2027.
- The revised plan of Thursday 28 January 2027 supersedes all earlier emails.
- The speaker briefing is on Friday 23 April 2027.
- The speaker programme checkpoint is on Friday 12 March 2027.
- The sponsorship kickoff can begin 3 working days after the audio-visual production sign-off is completed.
- The sponsorship kickoff is planned for Thursday 14 October 2027.
- The steering group approved an increase of £39,000 to the project budget in March.
- Ticket sales can begin 8 working days after venue contract signature ends.
- Venue contract signature is scheduled to end on Wednesday 3 February 2027.
- Working days are Monday to Friday; there are no public holidays in this period.
- Yuki Halloran owns the speaker programme workstream.

### Briefing

The Saltmarsh annual partner conference is now planned around the revised plan of Thursday 28 January 2027, which sets the conference opening day for Wednesday 14 July 2027 and supersedes all earlier emails. An earlier email of Thursday 14 January 2027 had given the opening day as Thursday 15 July 2027, so that date is retained here only as history. The dress rehearsal is on Thursday 12 August 2027, and the sponsorship kickoff is planned for Thursday 14 October 2027.

On ownership, Elif Sørensen owns the venue booking workstream, and Rohan Mensah supports it and attends its meetings. Yuki Halloran owns the speaker programme workstream, Noor Okafor owns audio-visual production, and Hamid Valdés owns sponsorship. Kwame Bakshi owns catering, with Lucía Esposito supporting that workstream and attending its meetings. Venue contract signature is scheduled to end on Wednesday 3 February 2027.

Several checkpoints fall in the spring. The speaker programme checkpoint is on Friday 12 March 2027, the registration review is on Thursday 8 April 2027, and the catering tasting is on Monday 19 April 2027. The speaker briefing follows on Friday 23 April 2027. Working days are Monday to Friday, and there are no public holidays in this period.

On sequencing, ticket sales can begin 8 working days after venue contract signature ends. The sponsorship kickoff can begin 3 working days after the audio-visual production sign-off is completed. The audio-visual production contract is worth £42,000.

On budget and capacity, the approved project budget is £106,000. The steering group approved an increase of £39,000 to the project budget in March, while a proposed further increase of £54,000 was rejected by the steering group. The plan provides for 526 delegate places. An additional 80 delegate places were approved in March, and a request for 30 more delegate places is still awaiting approval.

Four risks are being tracked. Late venue licensing could delay the conference opening day, and the keynote speaker's visa delay could do the same. A sponsor withdrawing could affect the budget, and low early registrations could affect the number of delegate places.

### Requirements and correct answers

R1. the project budget after all approved changes — **£145,000**
R2. the risks that could delay the conference opening day — **exactly these risks: late venue licensing; the keynote speaker's visa delay. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R3. the latest date the audio-visual production sign-off can be completed for the sponsorship kickoff to go ahead on its planned date — **Monday 11 October 2027**
R4. the earliest date ticket sales can begin — **Monday 15 February 2027**
R5. the value of the audio-visual production contract — **£42,000**
R6. who owns the catering workstream — **Kwame Bakshi**
R7. the current date of the conference opening day — **Wednesday 14 July 2027**
R8. who owns the venue booking workstream — **Elif Sørensen**
R9. the date of the catering tasting — **Monday 19 April 2027**
R10. how many working days after the speaker programme checkpoint the registration review takes place — **19**

### Reference memo

> **Memo to: Aisling Nakamura, Programme Director**
> **Re: Saltmarsh annual partner conference – status**
> 
> **Summary**
> 
> The conference is planned around the revised plan of Thursday 28 January 2027, which supersedes all earlier emails. The conference opens on Wednesday 14 July 2027. Budget, sequencing and ownership are set out below.
> 
> **Key facts**
> 
> - **Opening day:** The current date is Wednesday 14 July 2027. The earlier date of 15 July came from a superseded email.
> - **Budget:** The approved budget of £106,000 plus the approved £39,000 increase gives £145,000. The proposed £54,000 increase was rejected and is not included.
> - **Delay risks:** Two risks could delay the opening day: late venue licensing and the keynote speaker's visa delay.
> - **Ticket sales:** Venue contract signature ends on Wednesday 3 February 2027. Ticket sales can begin 8 working days later, so the earliest start is Monday 15 February 2027.
> - **Audio-visual sign-off:** The sponsorship kickoff is planned for Thursday 14 October 2027 and needs 3 working days after sign-off. The latest sign-off date is therefore Monday 11 October 2027.
> - **Audio-visual contract:** The contract is worth £42,000.
> - **Ownership:** Kwame Bakshi owns catering. Elif Sørensen owns venue booking.
> - **Catering tasting:** The tasting is on Monday 19 April 2027.
> - **Checkpoint spacing:** The registration review on Thursday 8 April 2027 falls 19 working days after the speaker programme checkpoint on Friday 12 March 2027.
> 
> **Next steps**
> 
> 1. Confirm the audio-visual sign-off is completed by Monday 11 October 2027.
> 2. Prepare to open ticket sales on Monday 15 February 2027.
> 3. Keep monitoring the licensing and visa risks, as they could move the opening day.
> 4. Please confirm that the budget of £145,000 is the figure to use for planning.

**Checks** (tick, or write the problem underneath):

- [x] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [x] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

PASS. Recalculated 19 working days, latest sign-off 11 October and earliest ticket sales 15 February. Budget is £106,000 + £39,000 = £145,000. Owners, contract, tasting date, revised opening date and the two opening risks agree.


---

## cal-s5-L3 — warehouse automation, level 3

### Fact sheet

- A proposed further increase of €8,000 was rejected by the steering group.
- A request for 2 more picking robots is still awaiting approval.
- An additional 2 picking robots were approved in March.
- An email of Monday 4 January 2027 gave the switch to automated picking as Friday 7 May 2027.
- Corentin Esposito supports the shift planning workstream, which Yuki Takahashi owns, and attends its meetings.
- Elif Fontaine owns the floor reinforcement workstream.
- Hamid Haddad owns the robot procurement workstream.
- Rafael Sørensen supports the maintenance contracts workstream, which Thandiwe Whitfield owns, and attends its meetings.
- Risk: a failed load test on the mezzanine could delay the switch to automated picking.
- Risk: delayed customs clearance for the robots could delay the switch to automated picking.
- Risk: spare-part price rises could affect the budget.
- Risk: union consultation on shift patterns could affect the number of picking robots.
- Robot installation can begin 4 working days after the floor reinforcement work ends.
- Sanna Oyelaran owns the software integration workstream.
- Thandiwe Whitfield owns the maintenance contracts workstream.
- The approved project budget is €195,000.
- The floor reinforcement kickoff can begin 3 working days after the software integration sign-off is completed.
- The floor reinforcement kickoff is planned for Wednesday 24 February 2027.
- The floor reinforcement work is scheduled to end on Thursday 25 February 2027.
- The insurance assessment is on Thursday 13 May 2027.
- The maintenance contract is worth €11,500.
- The operator training course is on Thursday 6 May 2027.
- The plan provides for 20 picking robots.
- The revised plan of Thursday 14 January 2027 sets the switch to automated picking for Monday 25 October 2027.
- The revised plan of Thursday 14 January 2027 supersedes all earlier emails.
- The robot procurement review is on Tuesday 11 May 2027.
- The safety training checkpoint is on Wednesday 21 April 2027.
- The steering group approved a reduction of €17,000 to the project budget in March.
- The supplier factory visit is on Wednesday 20 October 2027.
- Working days are Monday to Friday; there are no public holidays in this period.
- Yuki Takahashi owns the shift planning workstream.

### Briefing

The Granitefold warehouse automation is currently planned around the revised plan of Thursday 14 January 2027, which sets the switch to automated picking for Monday 25 October 2027. That plan supersedes all earlier emails, including the one of Monday 4 January 2027 that gave the switch as Friday 7 May 2027. The approved project budget is €195,000, and the plan provides for 20 picking robots.

On the budget side, the steering group approved a reduction of €17,000 to the project budget in March, while a proposed further increase of €8,000 was rejected by the steering group. An additional 2 picking robots were approved in March, and a separate request for 2 more picking robots is still awaiting approval. The maintenance contract is worth €11,500.

Ownership is spread across several people. Hamid Haddad owns the robot procurement workstream, and Sanna Oyelaran owns the software integration workstream. Elif Fontaine owns the floor reinforcement workstream. Yuki Takahashi owns the shift planning workstream, and Corentin Esposito supports it and attends its meetings. Thandiwe Whitfield owns the maintenance contracts workstream, with Rafael Sørensen supporting it and attending its meetings.

The floor reinforcement kickoff is planned for Wednesday 24 February 2027, and it can begin 3 working days after the software integration sign-off is completed. The floor reinforcement work is scheduled to end on Thursday 25 February 2027. Working days are Monday to Friday, and there are no public holidays in this period.

Robot installation can begin 4 working days after the floor reinforcement work ends. Several checkpoints fall in the months before the switch. The safety training checkpoint is on Wednesday 21 April 2027, followed by the operator training course on Thursday 6 May 2027. The robot procurement review is on Tuesday 11 May 2027, and the insurance assessment is on Thursday 13 May 2027. Later in the year, the supplier factory visit is on Wednesday 20 October 2027.

Four risks have been flagged. Delayed customs clearance for the robots could delay the switch to automated picking, and a failed load test on the mezzanine could do the same. Union consultation on shift patterns could affect the number of picking robots, and spare-part price rises could affect the budget.

### Requirements and correct answers

R1. the latest date the software integration sign-off can be completed for the floor reinforcement kickoff to go ahead on its planned date — **Friday 19 February 2027**
R2. the value of the maintenance contract — **€11,500**
R3. who owns the maintenance contracts workstream — **Thandiwe Whitfield**
R4. the project budget after all approved changes — **€178,000**
R5. the current date of the switch to automated picking — **Monday 25 October 2027**
R6. who owns the shift planning workstream — **Yuki Takahashi**
R7. the earliest date robot installation can begin — **Wednesday 3 March 2027**
R8. the date of the supplier factory visit — **Wednesday 20 October 2027**
R9. the risks that could delay the switch to automated picking — **exactly these risks: a failed load test on the mezzanine; delayed customs clearance for the robots. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R10. how many working days after the safety training checkpoint the robot procurement review takes place — **14**

### Reference memo

> **Memo to: Ingrid Quiroga, Programme Director**
> **Re: Granitefold warehouse automation status**
> 
> **Summary**
> 
> The programme is planned around the revised plan of Thursday 14 January 2027, which supersedes all earlier emails. The switch to automated picking is now set for Monday 25 October 2027. The project budget stands at €178,000 after the approved changes. Two risks could delay the switch: delayed customs clearance for the robots and a failed load test on the mezzanine.
> 
> **Key facts**
> 
> - **Switch date:** Monday 25 October 2027.
> - **Budget:** The approved €195,000 less the €17,000 reduction gives €178,000. The proposed €8,000 increase was rejected, so it does not count.
> - **Floor reinforcement kickoff:** Planned for Wednesday 24 February 2027. The software integration sign-off must be completed by Friday 19 February 2027 at the latest for this date to hold.
> - **Robot installation:** The earliest start is Wednesday 3 March 2027, four working days after the floor reinforcement work ends on Thursday 25 February 2027.
> - **Maintenance contract:** Worth €11,500. Thandiwe Whitfield owns the maintenance contracts workstream.
> - **Shift planning:** Yuki Takahashi owns this workstream.
> - **Supplier factory visit:** Wednesday 20 October 2027.
> - **Checkpoint spacing:** The robot procurement review on Tuesday 11 May 2027 falls 14 working days after the safety training checkpoint on Wednesday 21 April 2027.
> - **Delay risks:** Delayed customs clearance for the robots and a failed load test on the mezzanine could each delay the switch to automated picking.
> 
> **Next steps**
> 
> - Confirm that software integration sign-off will be completed by Friday 19 February 2027, so the floor reinforcement kickoff can go ahead as planned.
> - Track customs clearance for the robots and plan the mezzanine load test, as these are the two risks to the switch date.
> - Keep the 25 October 2027 switch date and the €178,000 budget as the working baseline for all workstream owners.

**Checks** (tick, or write the problem underneath):

- [x] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [x] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

PASS. Recalculated latest sign-off 19 February, earliest installation 3 March and a 14-working-day interval. Budget is €195,000 - €17,000 = €178,000. Ownership, contract, factory visit, current switch date and the two delay risks agree.


---

## cal-s6-L3 — conference, level 3

### Fact sheet

- A proposed further increase of £37,000 was rejected by the steering group.
- A request for 12 more delegate places is still awaiting approval.
- Aisling Novak supports the venue booking workstream, which Noor Quiroga owns, and attends its meetings.
- Amara Sørensen owns the catering workstream.
- An additional 24 delegate places were approved in March.
- An email of Thursday 7 January 2027 gave the conference opening day as Friday 1 October 2027.
- Idris Demir owns the speaker programme workstream.
- Kwame Delacroix owns the audio-visual production workstream.
- Noor Quiroga owns the venue booking workstream.
- Oluwaseun Okafor supports the speaker programme workstream, which Idris Demir owns, and attends its meetings.
- Risk: a sponsor withdrawing could affect the budget.
- Risk: late venue licensing could delay the conference opening day.
- Risk: low early registrations could affect the number of delegate places.
- Risk: the keynote speaker's visa delay could delay the conference opening day.
- Svetlana Lindqvist owns the registration workstream.
- The approved project budget is £283,000.
- The audio-visual production contract is worth £42,500.
- The catering checkpoint is on Wednesday 3 February 2027.
- The catering tasting is on Thursday 8 July 2027.
- The plan provides for 163 delegate places.
- The registration kickoff can begin 6 working days after the audio-visual production sign-off is completed.
- The registration kickoff is planned for Wednesday 11 August 2027.
- The revised plan of Thursday 28 January 2027 sets the conference opening day for Monday 15 November 2027.
- The revised plan of Thursday 28 January 2027 supersedes all earlier emails.
- The speaker briefing is on Monday 14 June 2027.
- The sponsor call is on Thursday 11 March 2027.
- The sponsorship review is on Thursday 18 February 2027.
- The steering group approved an increase of £43,000 to the project budget in March.
- Ticket sales can begin 10 working days after venue contract signature ends.
- Venue contract signature is scheduled to end on Monday 28 June 2027.
- Working days are Monday to Friday; there are no public holidays in this period.

### Briefing

The Ostrava annual partner conference is currently planned around a conference opening day of Monday 15 November 2027, as set by the revised plan of Thursday 28 January 2027. That plan supersedes all earlier emails, including the one of Thursday 7 January 2027 that gave the opening day as Friday 1 October 2027. The approved project budget is £283,000, and the steering group approved an increase of £43,000 to the project budget in March. A proposed further increase of £37,000 was rejected by the steering group.

On ownership, Noor Quiroga owns the venue booking workstream, with Aisling Novak supporting it and attending its meetings. Idris Demir owns the speaker programme workstream, and Oluwaseun Okafor supports it and attends its meetings. Svetlana Lindqvist owns the registration workstream, Kwame Delacroix owns the audio-visual production workstream, and Amara Sørensen owns the catering workstream. The audio-visual production contract is worth £42,500.

Several milestones are already in the calendar. The catering checkpoint is on Wednesday 3 February 2027, followed by the sponsorship review on Thursday 18 February 2027 and the sponsor call on Thursday 11 March 2027. The speaker briefing is on Monday 14 June 2027, and the catering tasting is on Thursday 8 July 2027. Venue contract signature is scheduled to end on Monday 28 June 2027.

Registration and ticketing depend on earlier steps. The registration kickoff is planned for Wednesday 11 August 2027, and it can begin 6 working days after the audio-visual production sign-off is completed. Ticket sales can begin 10 working days after venue contract signature ends. Working days are Monday to Friday, and there are no public holidays in this period.

On capacity, the plan provides for 163 delegate places. An additional 24 delegate places were approved in March, while a request for 12 more delegate places is still awaiting approval.

Three risks to the conference opening day are recorded, along with others that touch capacity and budget. Late venue licensing could delay the conference opening day, and the keynote speaker's visa delay could do the same. Low early registrations could affect the number of delegate places, and a sponsor withdrawing could affect the budget.

### Requirements and correct answers

R1. the date of the catering tasting — **Thursday 8 July 2027**
R2. the latest date the audio-visual production sign-off can be completed for the registration kickoff to go ahead on its planned date — **Tuesday 3 August 2027**
R3. who owns the venue booking workstream — **Noor Quiroga**
R4. the earliest date ticket sales can begin — **Monday 12 July 2027**
R5. how many working days after the catering checkpoint the sponsorship review takes place — **11**
R6. who owns the speaker programme workstream — **Idris Demir**
R7. the project budget after all approved changes — **£326,000**
R8. the current date of the conference opening day — **Monday 15 November 2027**
R9. the risks that could delay the conference opening day — **exactly these risks: the keynote speaker's visa delay; late venue licensing. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R10. the value of the audio-visual production contract — **£42,500**

### Reference memo

> **Memo to Sanna Nakamura, Programme Director**
> 
> **Summary**
> 
> The Ostrava annual partner conference is now planned to open on Monday 15 November 2027. This is set by the revised plan of Thursday 28 January 2027, which supersedes all earlier emails, including the one giving 1 October. The project budget stands at £326,000 after all approved changes. The registration kickoff on Wednesday 11 August 2027 remains achievable if the audio-visual sign-off is on time.
> 
> **Key facts**
> 
> - **Catering tasting:** Thursday 8 July 2027.
> - **Audio-visual sign-off deadline:** Tuesday 3 August 2027 is the latest date it can be completed for the registration kickoff to go ahead as planned.
> - **Venue booking owner:** Noor Quiroga.
> - **Ticket sales:** the earliest start is Monday 12 July 2027.
> - **Sponsorship review:** it takes place 11 working days after the catering checkpoint.
> - **Speaker programme owner:** Idris Demir.
> - **Project budget:** £326,000 after all approved changes.
> - **Conference opening day:** currently Monday 15 November 2027.
> - **Risks to the opening day:** two recorded risks could delay it. They are late venue licensing and the keynote speaker's visa delay.
> - **Audio-visual production contract:** £42,500.
> 
> **Next steps**
> 
> 1. Ask Kwame Delacroix to confirm that the audio-visual sign-off will be completed by Tuesday 3 August 2027.
> 2. Ask Noor Quiroga to monitor the venue licensing position.
> 3. Ask Idris Demir to monitor the keynote speaker's visa position.
> 4. Use 15 November 2027 as the only opening date in all communications.

**Checks** (tick, or write the problem underneath):

- [ ] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [ ] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [ ] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

FAIL — contradictory risk count. The briefing says "Three risks to the conference opening day are recorded", but the fact sheet has only TWO: late venue licensing and the keynote speaker's visa delay. The remaining two risks concern capacity and budget. Replace that sentence with "Two risks to the conference opening day are recorded, alongside two risks to capacity and budget." Until corrected, R9 is not unambiguous from the briefing alone: its claim of a third opening risk conflicts with the supplied exact-two answer. Checks 1–3 are left unticked. The reference correctly follows the fact sheet and names two opening risks. All numeric/date answers independently agree: latest sign-off 3 August; earliest ticket sales 12 July; interval 11 working days; budget £326,000.


---

## cal-s7-L3 — school IT refresh, level 3

### Fact sheet

- A proposed further increase of £6,000 was rejected by the steering group.
- A request for 12 more laptops is still awaiting approval.
- Aisling Sørensen owns the parent communications workstream.
- An additional 21 laptops were approved in March.
- An email of Thursday 7 January 2027 gave the first lessons on the new devices as Tuesday 19 October 2027.
- Device deployment can begin 5 working days after the network upgrade ends.
- Emeka Abernethy supports the parent communications workstream, which Aisling Sørensen owns, and attends its meetings.
- Freya Haddad owns the device procurement workstream.
- Kwame Lindqvist owns the network upgrade workstream.
- Priya Okafor owns the asset tagging workstream.
- Rafael Halloran supports the device procurement workstream, which Freya Haddad owns, and attends its meetings.
- Risk: a shortage of charging trolleys could delay the first lessons on the new devices.
- Risk: device breakage rates could affect the number of laptops.
- Risk: higher insurance costs could affect the budget.
- Risk: late approval of the content filter could delay the first lessons on the new devices.
- Rohan Nakamura owns the safeguarding filters workstream.
- The approved project budget is £429,000.
- The asset tagging kickoff can begin 6 working days after the network upgrade sign-off is completed.
- The asset tagging kickoff is planned for Thursday 6 May 2027.
- The device leasing agreement is worth £64,000.
- The governors' meeting is on Monday 11 October 2027.
- The network upgrade is scheduled to end on Tuesday 16 February 2027.
- The parents' evening is on Monday 8 February 2027.
- The plan provides for 187 laptops.
- The revised plan of Monday 18 January 2027 sets the first lessons on the new devices for Friday 9 April 2027.
- The revised plan of Monday 18 January 2027 supersedes all earlier emails.
- The safeguarding filters review is on Wednesday 19 May 2027.
- The safeguarding review is on Monday 22 February 2027.
- The steering group approved an increase of £11,000 to the project budget in March.
- The teacher training checkpoint is on Monday 19 April 2027.
- Working days are Monday to Friday; there are no public holidays in this period.

### Briefing

The Bramblecote classroom technology refresh is working to the revised plan of Monday 18 January 2027, which supersedes all earlier emails. That plan sets the first lessons on the new devices for Friday 9 April 2027. An email of Thursday 7 January 2027 had given the first lessons as Tuesday 19 October 2027, but the revised plan is the governing document. The plan provides for 187 laptops, and an additional 21 laptops were approved in March. A request for 12 more laptops is still awaiting approval.

On budget, the approved project budget is £429,000. The steering group approved an increase of £11,000 to the project budget in March, while a proposed further increase of £6,000 was rejected. Separately, the device leasing agreement is worth £64,000. Higher insurance costs are flagged as a risk that could affect the budget.

Workstream ownership is settled. Kwame Lindqvist owns the network upgrade, which is scheduled to end on Tuesday 16 February 2027. Freya Haddad owns device procurement, and Rafael Halloran supports that workstream and attends its meetings. Aisling Sørensen owns parent communications, with Emeka Abernethy supporting and attending its meetings. Rohan Nakamura owns the safeguarding filters workstream, and Priya Okafor owns asset tagging.

Sequencing depends on the network work. Device deployment can begin 5 working days after the network upgrade ends, and the asset tagging kickoff can begin 6 working days after the network upgrade sign-off is completed. Working days are Monday to Friday, and there are no public holidays in this period. The asset tagging kickoff itself is planned for Thursday 6 May 2027.

Several review and engagement dates sit around the delivery work. The parents' evening is on Monday 8 February 2027, and the safeguarding review is on Monday 22 February 2027. The teacher training checkpoint is on Monday 19 April 2027, and the safeguarding filters review is on Wednesday 19 May 2027. The governors' meeting is on Monday 11 October 2027.

Four risks are on the register. A shortage of charging trolleys could delay the first lessons on the new devices, and late approval of the content filter could do the same. Device breakage rates could affect the number of laptops. Higher insurance costs, as noted above, could affect the budget.

### Requirements and correct answers

R1. the risks that could delay the first lessons on the new devices — **exactly these risks: late approval of the content filter; a shortage of charging trolleys. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R2. the current date of the first lessons on the new devices — **Friday 9 April 2027**
R3. how many working days after the teacher training checkpoint the safeguarding filters review takes place — **22**
R4. the date of the safeguarding review — **Monday 22 February 2027**
R5. who owns the device procurement workstream — **Freya Haddad**
R6. the value of the device leasing agreement — **£64,000**
R7. the latest date the network upgrade sign-off can be completed for the asset tagging kickoff to go ahead on its planned date — **Wednesday 28 April 2027**
R8. the earliest date device deployment can begin — **Tuesday 23 February 2027**
R9. the project budget after all approved changes — **£440,000**
R10. who owns the parent communications workstream — **Aisling Sørensen**

### Reference memo

> **Memo to: Oluwaseun Mensah, Programme Director**
> **Subject: Bramblecote classroom technology refresh – status**
> 
> **Summary**
> 
> The refresh is working to the revised plan of Monday 18 January 2027, which supersedes all earlier emails. The first lessons on the new devices are now set for Friday 9 April 2027. Two risks could delay them: a shortage of charging trolleys and late approval of the content filter.
> 
> **Key facts**
> 
> - **First lessons:** Friday 9 April 2027. The earlier date of 19 October 2027 is superseded.
> - **Delay risks:** A shortage of charging trolleys and late approval of the content filter are the only two risks that could delay the first lessons.
> - **Safeguarding review:** Monday 22 February 2027.
> - **Safeguarding filters review:** Wednesday 19 May 2027, which is 22 working days after the teacher training checkpoint on Monday 19 April 2027.
> - **Device procurement owner:** Freya Haddad.
> - **Parent communications owner:** Aisling Sørensen.
> - **Device leasing agreement:** £64,000.
> - **Project budget after approved changes:** £440,000, which is the original £429,000 plus the £11,000 increase approved in March. The further £6,000 increase was rejected.
> - **Earliest device deployment:** Tuesday 23 February 2027, five working days after the network upgrade ends on Tuesday 16 February 2027.
> - **Asset tagging kickoff:** planned for Thursday 6 May 2027. Network upgrade sign-off must therefore be completed by Wednesday 28 April 2027 at the latest, as the kickoff can begin 6 working days after sign-off.
> 
> **Next steps**
> 
> 1. Monitor charging trolley supply and the content filter approval, as both could affect the 9 April date.
> 2. Keep the network upgrade on track to finish on 16 February, and complete its sign-off no later than 28 April.
> 3. Track the insurance cost risk to the budget, and the breakage risk to laptop numbers.
> 4. Chase the pending request for 12 more laptops.

**Checks** (tick, or write the problem underneath):

- [x] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [x] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

PASS. The safeguarding review (22 February) is distinct from the safeguarding filters review (19 May), and both are correctly attached. Recalculated interval 22 working days, latest sign-off 28 April and earliest deployment 23 February. Budget £440,000, ownership and exact delay risks agree.


---

## cal-s8-L3 — conference, level 3

### Fact sheet

- A proposed further increase of £15,000 was rejected by the steering group.
- A request for 31 more delegate places is still awaiting approval.
- An additional 39 delegate places were approved in March.
- An email of Friday 8 January 2027 gave the conference opening day as Monday 27 September 2027.
- Corentin Takahashi supports the audio-visual production workstream, which Oluwaseun Quiroga owns, and attends its meetings.
- Emeka Achterberg owns the registration workstream.
- Ingrid Iyer owns the catering workstream.
- Lucía Whitfield owns the sponsorship workstream.
- Mei Valdés owns the speaker programme workstream.
- Oluwaseun Quiroga owns the audio-visual production workstream.
- Risk: late venue licensing could delay the conference opening day.
- Risk: low early registrations could affect the number of delegate places.
- Risk: rising catering prices could affect the budget.
- Risk: the keynote speaker's visa delay could delay the conference opening day.
- The approved project budget is £111,000.
- The audio-visual production contract is worth £52,000.
- The catering kickoff can begin 8 working days after the speaker programme sign-off is completed.
- The catering kickoff is planned for Friday 15 October 2027.
- The catering tasting is on Thursday 25 March 2027.
- The plan provides for 788 delegate places.
- The registration desk training is on Thursday 15 April 2027.
- The revised plan of Friday 29 January 2027 sets the conference opening day for Tuesday 27 July 2027.
- The revised plan of Friday 29 January 2027 supersedes all earlier emails.
- The speaker briefing is on Tuesday 11 May 2027.
- The sponsorship review is on Friday 16 April 2027.
- The steering group approved an increase of £59,000 to the project budget in March.
- The venue booking checkpoint is on Tuesday 23 March 2027.
- Ticket sales can begin 8 working days after venue contract signature ends.
- Venue contract signature is scheduled to end on Tuesday 15 June 2027.
- Working days are Monday to Friday; there are no public holidays in this period.
- Yuki Novak supports the registration workstream, which Emeka Achterberg owns, and attends its meetings.

### Briefing

The Bramblecote annual partner conference is being planned around a revised plan dated Friday 29 January 2027, which sets the conference opening day for Tuesday 27 July 2027. That plan supersedes all earlier emails, including the one of Friday 8 January 2027 that had given the opening day as Monday 27 September 2027. Working days are Monday to Friday, and there are no public holidays in this period.

On budget, the approved project budget is £111,000. The steering group approved an increase of £59,000 in March, while a proposed further increase of £15,000 was rejected. The audio-visual production contract is worth £52,000. One risk is that rising catering prices could affect the budget.

Delegate numbers follow a similar pattern. The plan provides for 788 delegate places, and an additional 39 delegate places were approved in March. A request for 31 more delegate places is still awaiting approval. A further risk is that low early registrations could affect the number of delegate places.

Workstream ownership is settled. Emeka Achterberg owns the registration workstream, and Yuki Novak supports it and attends its meetings. Oluwaseun Quiroga owns the audio-visual production workstream, with Corentin Takahashi supporting it and attending its meetings. Lucía Whitfield owns the sponsorship workstream, Ingrid Iyer owns the catering workstream, and Mei Valdés owns the speaker programme workstream.

Several checkpoints are already in the calendar. The venue booking checkpoint is on Tuesday 23 March 2027, followed by the catering tasting on Thursday 25 March 2027. The registration desk training is on Thursday 15 April 2027, and the sponsorship review is on Friday 16 April 2027. The speaker briefing is on Tuesday 11 May 2027. Venue contract signature is scheduled to end on Tuesday 15 June 2027.

Dependencies between milestones are expressed as working-day gaps. Ticket sales can begin 8 working days after venue contract signature ends. The catering kickoff can begin 8 working days after the speaker programme sign-off is completed, and it is planned for Friday 15 October 2027.

Two further risks concern the opening day itself. Late venue licensing could delay the conference opening day, and the keynote speaker's visa delay could also delay it.

### Requirements and correct answers

R1. the earliest date ticket sales can begin — **Friday 25 June 2027**
R2. the risks that could delay the conference opening day — **exactly these risks: late venue licensing; the keynote speaker's visa delay. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R3. how many working days after the venue booking checkpoint the sponsorship review takes place — **18**
R4. the value of the audio-visual production contract — **£52,000**
R5. the current date of the conference opening day — **Tuesday 27 July 2027**
R6. the latest date the speaker programme sign-off can be completed for the catering kickoff to go ahead on its planned date — **Tuesday 5 October 2027**
R7. the date of the speaker briefing — **Tuesday 11 May 2027**
R8. who owns the registration workstream — **Emeka Achterberg**
R9. the project budget after all approved changes — **£170,000**
R10. who owns the audio-visual production workstream — **Oluwaseun Quiroga**

### Reference memo

> **Memo to: Sanna Delacroix, Programme Director**
> **Re: Bramblecote annual partner conference – status**
> 
> **Summary**
> 
> The conference is planned around the revised plan of Friday 29 January 2027, which supersedes all earlier emails. The opening day is now Tuesday 27 July 2027. Two risks could delay that date: late venue licensing and the keynote speaker's visa delay.
> 
> **Key facts**
> 
> - **Earliest ticket sales start:** Friday 25 June 2027, which is 8 working days after venue contract signature ends on Tuesday 15 June 2027.
> - **Opening day risks:** late venue licensing and the keynote speaker's visa delay.
> - **Sponsorship review:** Friday 16 April 2027, which is 18 working days after the venue booking checkpoint on Tuesday 23 March 2027.
> - **Audio-visual production contract:** £52,000.
> - **Current opening day:** Tuesday 27 July 2027.
> - **Latest speaker programme sign-off:** Tuesday 5 October 2027, for the catering kickoff to go ahead on its planned date of Friday 15 October 2027.
> - **Speaker briefing:** Tuesday 11 May 2027.
> - **Registration workstream owner:** Emeka Achterberg.
> - **Project budget after approved changes:** £170,000, being the £111,000 approved budget plus the £59,000 increase approved in March. The proposed further £15,000 was rejected.
> - **Audio-visual production workstream owner:** Oluwaseun Quiroga.
> 
> **Next steps**
> 
> 1. Plan all milestones against the opening day of Tuesday 27 July 2027, and disregard the earlier September date.
> 2. Hold ticket sales until Friday 25 June 2027 at the earliest.
> 3. Make sure the speaker programme sign-off is completed by Tuesday 5 October 2027.
> 4. Track the venue licensing and keynote visa risks, since each could move the opening day.

**Checks** (tick, or write the problem underneath):

- [x] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [x] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

PASS. Recalculated earliest ticket sales 25 June, checkpoint interval 18 working days and latest sign-off 5 October. Budget is £111,000 + £59,000 = £170,000. Ownership, contract, speaker briefing, revised opening date and exact risk set agree.


---

## cal-s9-L3 — school IT refresh, level 3

### Fact sheet

- A proposed further increase of £24,000 was rejected by the steering group.
- A request for 20 more laptops is still awaiting approval.
- An additional 42 laptops were approved in March.
- An email of Tuesday 19 January 2027 gave the first lessons on the new devices as Friday 27 August 2027.
- Anders Novak supports the network upgrade workstream, which Dmitri Bakshi owns, and attends its meetings.
- Device deployment can begin 4 working days after the network upgrade ends.
- Dmitri Bakshi owns the network upgrade workstream.
- Elif Okafor supports the asset tagging workstream, which Noor Fontaine owns, and attends its meetings.
- Emeka Kowalczyk owns the safeguarding filters workstream.
- Mei Esposito owns the device procurement workstream.
- Noor Fontaine owns the asset tagging workstream.
- Risk: delays in the wireless survey could delay the first lessons on the new devices.
- Risk: higher insurance costs could affect the number of laptops.
- Risk: late approval of the content filter could delay the first lessons on the new devices.
- Risk: rising support fees could affect the budget.
- Thandiwe Achterberg owns the parent communications workstream.
- The approved project budget is £354,000.
- The device leasing agreement is worth £72,000.
- The governors' meeting is on Monday 18 October 2027.
- The network upgrade is scheduled to end on Thursday 17 June 2027.
- The parent communications review is on Friday 25 June 2027.
- The plan provides for 279 laptops.
- The revised plan of Friday 22 January 2027 sets the first lessons on the new devices for Thursday 26 August 2027.
- The revised plan of Friday 22 January 2027 supersedes all earlier emails.
- The safeguarding filters kickoff can begin 8 working days after the device procurement sign-off is completed.
- The safeguarding filters kickoff is planned for Monday 28 June 2027.
- The safeguarding review is on Thursday 11 February 2027.
- The steering group approved an increase of £14,000 to the project budget in March.
- The supplier demo is on Monday 8 November 2027.
- The teacher training checkpoint is on Thursday 3 June 2027.
- Working days are Monday to Friday; there are no public holidays in this period.

### Briefing

The Tidewell classroom technology refresh now has a clear set of workstream owners. Mei Esposito owns device procurement, Dmitri Bakshi owns the network upgrade, Noor Fontaine owns asset tagging, Emeka Kowalczyk owns safeguarding filters, and Thandiwe Achterberg owns parent communications. Anders Novak supports the network upgrade workstream, which Dmitri Bakshi owns, and attends its meetings. Elif Okafor does the same for asset tagging, which Noor Fontaine owns.

On timing, there is a disagreement between sources about the first lessons on the new devices. An email of Tuesday 19 January 2027 gave the date as Friday 27 August 2027, whereas the revised plan of Friday 22 January 2027 sets it for Thursday 26 August 2027. The revised plan supersedes all earlier emails. The network upgrade is scheduled to end on Thursday 17 June 2027.

Device deployment can begin 4 working days after the network upgrade ends. Working days are Monday to Friday, and there are no public holidays in this period. The safeguarding filters kickoff is planned for Monday 28 June 2027, and it can begin 8 working days after the device procurement sign-off is completed.

On money and equipment, the approved project budget is £354,000, and the device leasing agreement is worth £72,000. In March the steering group approved an increase of £14,000 to the project budget. A proposed further increase of £24,000 was rejected by the steering group. The plan provides for 279 laptops, and an additional 42 laptops were approved in March. A request for 20 more laptops is still awaiting approval.

Several milestones and meetings are on the calendar. The safeguarding review is on Thursday 11 February 2027, the teacher training checkpoint is on Thursday 3 June 2027, and the parent communications review is on Friday 25 June 2027. The governors' meeting is on Monday 18 October 2027, and the supplier demo is on Monday 8 November 2027.

Four risks are being tracked. Late approval of the content filter could delay the first lessons on the new devices, and delays in the wireless survey could do the same. Higher insurance costs could affect the number of laptops, and rising support fees could affect the budget.

### Requirements and correct answers

R1. the current date of the first lessons on the new devices — **Thursday 26 August 2027**
R2. who owns the network upgrade workstream — **Dmitri Bakshi**
R3. the earliest date device deployment can begin — **Wednesday 23 June 2027**
R4. who owns the asset tagging workstream — **Noor Fontaine**
R5. the risks that could delay the first lessons on the new devices — **exactly these risks: delays in the wireless survey; late approval of the content filter. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.**
R6. the value of the device leasing agreement — **£72,000**
R7. the latest date the device procurement sign-off can be completed for the safeguarding filters kickoff to go ahead on its planned date — **Wednesday 16 June 2027**
R8. the date of the supplier demo — **Monday 8 November 2027**
R9. the project budget after all approved changes — **£368,000**
R10. how many working days after the teacher training checkpoint the parent communications review takes place — **16**

### Reference memo

> **Memo to: Joaquin Petrov, Programme Director**
> **Re: Tidewell classroom technology refresh – status**
> 
> **Summary**
> 
> The programme has clear workstream owners, and the first lessons on the new devices are now set for Thursday 26 August 2027. This date comes from the revised plan of 22 January 2027, which supersedes the earlier email that gave 27 August. Two risks could delay those lessons: late approval of the content filter and delays in the wireless survey.
> 
> **Key facts**
> 
> - **First lessons:** Thursday 26 August 2027.
> - **Network upgrade owner:** Dmitri Bakshi. The upgrade ends on Thursday 17 June 2027.
> - **Earliest device deployment:** Wednesday 23 June 2027, four working days after the network upgrade ends.
> - **Asset tagging owner:** Noor Fontaine.
> - **Risks to the first lessons:** delays in the wireless survey and late approval of the content filter.
> - **Device leasing agreement:** £72,000.
> - **Latest device procurement sign-off:** Wednesday 16 June 2027. The safeguarding filters kickoff is planned for Monday 28 June 2027 and needs eight working days after sign-off.
> - **Supplier demo:** Monday 8 November 2027.
> - **Project budget after approved changes:** £368,000. This is the £354,000 approved budget plus the £14,000 increase approved in March. The further £24,000 increase was rejected.
> - **Teacher training checkpoint to parent communications review:** the review on Friday 25 June 2027 falls 16 working days after the checkpoint on Thursday 3 June 2027.
> 
> **Next steps**
> 
> 1. Hold device procurement sign-off no later than Wednesday 16 June 2027 so the filters kickoff can go ahead on 28 June.
> 2. Track the wireless survey and the content filter approval closely, as both could affect the first lessons.
> 3. Plan device deployment to start from Wednesday 23 June 2027, once the network upgrade has finished.

**Checks** (tick, or write the problem underneath):

- [x] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [x] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [x] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [x] The reference memo is correct.

Notes:

PASS. Recalculated earliest deployment 23 June, latest sign-off 16 June and a 16-working-day interval. Budget is £354,000 + £14,000 = £368,000. Owner/supporter distinctions, revised lessons date, lease, supplier demo and exact risk set agree.
