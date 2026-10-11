# Style pairs: check content is identical (DESIGN.md §6.4)

For each pair, check that the warm and plain versions state exactly the same facts as the original (including any mistakes). Note any difference under the pair.


---

## S01

**Original**

```text
**Memo to Idris Demir, Programme Director**
**Subject: Marrowfield office relocation: status update**

**Summary**

The relocation is running to a budget of £365,000, and the official opening is currently set for Thursday 15 April 2027. Two risks could delay that opening, and the key dates and owners are set out below.

**Key facts**

- **Budget review:** Thursday 3 June 2027.
- **Project budget:** £365,000 after all approved changes (£350,000 plus £39,000, less £24,000). The proposed £18,000 increase was rejected and is not included.
- **Risks to the opening:** A dispute over the lift installation and a delay in fire-safety certification could each delay the official opening.
- **Official opening:** Thursday 15 April 2027, per the revised plan of 21 January 2027. This takes precedence over the earlier emails, which gave Tuesday 8 June 2027.
- **Furniture procurement review:** Monday 4 October 2027, which is 18 working days after the security badges checkpoint on Wednesday 8 September 2027.
- **Staff move-in:** It can begin on Wednesday 10 March 2027 at the earliest. Cabling ends on Wednesday 3 March, 5 working days after the landlord handover on Wednesday 24 February.
- **Staff communications:** Owned by Yuki Kowalczyk.
- **Furniture supplier contract:** Worth £81,000.
- **Lease negotiation:** Owned by Amara Castellanos.
- **Network cabling sign-off:** To keep the moving logistics kickoff on its planned date of Wednesday 23 June 2027, sign-off must be completed by Thursday 17 June 2027 at the latest.

**Next steps**

1. Monitor the lift installation dispute and the fire-safety certification, as both threaten the opening date.
2. Prepare for the budget review on Thursday 3 June 2027, using the £365,000 figure.
3. Confirm the cabling sign-off timetable so that the moving logistics kickoff can proceed as planned.
```

**Warm**

```text
**Memo to Idris Demir, Programme Director**
**Subject: Marrowfield office relocation: status update**

**Summary**

Good news to share: the relocation is running to a budget of £365,000, and the official opening is currently set for Thursday 15 April 2027. Two risks could delay that opening, and we've set out the key dates and owners below for you.

**Key facts**

- **Budget review:** Thursday 3 June 2027.
- **Project budget:** £365,000 after all approved changes (£350,000 plus £39,000, less £24,000). The proposed £18,000 increase was rejected and is not included.
- **Risks to the opening:** A dispute over the lift installation and a delay in fire-safety certification could each delay the official opening.
- **Official opening:** Thursday 15 April 2027, per the revised plan of 21 January 2027. This takes precedence over the earlier emails, which gave Tuesday 8 June 2027.
- **Furniture procurement review:** Monday 4 October 2027, which is 18 working days after the security badges checkpoint on Wednesday 8 September 2027.
- **Staff move-in:** It can begin on Wednesday 10 March 2027 at the earliest. Cabling ends on Wednesday 3 March, 5 working days after the landlord handover on Wednesday 24 February.
- **Staff communications:** Owned by Yuki Kowalczyk.
- **Furniture supplier contract:** Worth £81,000.
- **Lease negotiation:** Owned by Amara Castellanos.
- **Network cabling sign-off:** To keep the moving logistics kickoff on its planned date of Wednesday 23 June 2027, sign-off must be completed by Thursday 17 June 2027 at the latest.

**Next steps**

1. Please keep an eye on the lift installation dispute and the fire-safety certification, as both threaten the opening date.
2. Let's prepare for the budget review on Thursday 3 June 2027, using the £365,000 figure.
3. Let's confirm the cabling sign-off timetable so that the moving logistics kickoff can proceed as planned.
```

**Plain**

```text
**Memo to Idris Demir, Programme Director**
**Subject: Marrowfield office relocation: status update**

**Summary**

The relocation is running to a budget of £365,000. The official opening is set for Thursday 15 April 2027. Two risks could delay the opening. Key dates and owners are listed below.

**Key facts**

- **Budget review:** Thursday 3 June 2027.
- **Project budget:** £365,000 after all approved changes (£350,000 plus £39,000, less £24,000). The proposed £18,000 increase was rejected and is not included.
- **Risks to the opening:** A dispute over the lift installation and a delay in fire-safety certification could each delay the official opening.
- **Official opening:** Thursday 15 April 2027, per the revised plan of 21 January 2027. This takes precedence over the earlier emails, which gave Tuesday 8 June 2027.
- **Furniture procurement review:** Monday 4 October 2027, which is 18 working days after the security badges checkpoint on Wednesday 8 September 2027.
- **Staff move-in:** It can begin on Wednesday 10 March 2027 at the earliest. Cabling ends on Wednesday 3 March, 5 working days after the landlord handover on Wednesday 24 February.
- **Staff communications:** Owned by Yuki Kowalczyk.
- **Furniture supplier contract:** Worth £81,000.
- **Lease negotiation:** Owned by Amara Castellanos.
- **Network cabling sign-off:** To keep the moving logistics kickoff on its planned date of Wednesday 23 June 2027, sign-off must be completed by Thursday 17 June 2027 at the latest.

**Next steps**

1. Monitor the lift installation dispute and the fire-safety certification. Both threaten the opening date.
2. Prepare for the budget review on Thursday 3 June 2027, using the £365,000 figure.
3. Confirm the cabling sign-off timetable so the moving logistics kickoff can proceed as planned.
```

Requirements: R1 the date of the budget review = Thursday 3 June 2027; R2 the project budget after all approved changes = £365,000; R3 the risks that could delay the official opening of the new office = exactly these risks: a dispute over the lift installation; a delay in fire-safety certification. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R4 the current date of the official opening of the new office = Thursday 15 April 2027; R5 how many working days after the security badges checkpoint the furniture procurement review takes place = 18; R6 the earliest date the staff move-in can begin = Wednesday 10 March 2027; R7 who owns the staff communications workstream = Yuki Kowalczyk; R8 the value of the furniture supplier contract = £81,000; R9 who owns the lease negotiation workstream = Amara Castellanos; R10 the latest date the network cabling sign-off can be completed for the moving logistics kickoff to go ahead on its planned date = Thursday 17 June 2027

Notes:


---

## S02

**Original**

```text
**Memo to Sanna Nakamura, Programme Director**

**Summary**

The Ostrava annual partner conference is now planned to open on Monday 15 November 2027. This is set by the revised plan of Thursday 28 January 2027, which supersedes all earlier emails, including the one giving 1 October. The project budget stands at £326,000 after all approved changes. The registration kickoff on Wednesday 11 August 2027 remains achievable if the audio-visual sign-off is on time.

**Key facts**

- **Catering tasting:** Thursday 8 July 2027.
- **Audio-visual sign-off deadline:** Tuesday 3 August 2027 is the latest date it can be completed for the registration kickoff to go ahead as planned.
- **Venue booking owner:** Noor Quiroga.
- **Ticket sales:** the earliest start is Monday 12 July 2027.
- **Sponsorship review:** it takes place 11 working days after the catering checkpoint.
- **Speaker programme owner:** Idris Demir.
- **Project budget:** £326,000 after all approved changes.
- **Conference opening day:** currently Monday 15 November 2027.
- **Risks to the opening day:** two recorded risks could delay it. They are late venue licensing and the keynote speaker's visa delay.
- **Audio-visual production contract:** £42,500.

**Next steps**

1. Ask Kwame Delacroix to confirm that the audio-visual sign-off will be completed by Tuesday 3 August 2027.
2. Ask Noor Quiroga to monitor the venue licensing position.
3. Ask Idris Demir to monitor the keynote speaker's visa position.
4. Use 15 November 2027 as the only opening date in all communications.
```

**Warm**

```text
**Memo to Sanna Nakamura, Programme Director**

**Summary**

Good news on the Ostrava annual partner conference: it is now planned to open on Monday 15 November 2027. This is set by the revised plan of Thursday 28 January 2027, which supersedes all earlier emails, including the one giving 1 October. The project budget stands at £326,000 after all approved changes. The registration kickoff on Wednesday 11 August 2027 remains achievable if the audio-visual sign-off is on time.

**Key facts**

- **Catering tasting:** Thursday 8 July 2027.
- **Audio-visual sign-off deadline:** Tuesday 3 August 2027 is the latest date it can be completed for the registration kickoff to go ahead as planned.
- **Venue booking owner:** Noor Quiroga.
- **Ticket sales:** the earliest start is Monday 12 July 2027.
- **Sponsorship review:** it takes place 11 working days after the catering checkpoint.
- **Speaker programme owner:** Idris Demir.
- **Project budget:** £326,000 after all approved changes.
- **Conference opening day:** currently Monday 15 November 2027.
- **Risks to the opening day:** two recorded risks could delay it. They are late venue licensing and the keynote speaker's visa delay.
- **Audio-visual production contract:** £42,500.

**Next steps**

1. Please ask Kwame Delacroix to confirm that the audio-visual sign-off will be completed by Tuesday 3 August 2027.
2. Please ask Noor Quiroga to keep an eye on the venue licensing position.
3. Please ask Idris Demir to keep an eye on the keynote speaker's visa position.
4. Let's use 15 November 2027 as the only opening date in all communications.
```

**Plain**

```text
**Memo to Sanna Nakamura, Programme Director**

**Summary**

The Ostrava annual partner conference is planned to open on Monday 15 November 2027. The revised plan of Thursday 28 January 2027 sets this date and supersedes all earlier emails, including the one giving 1 October. The project budget is £326,000 after all approved changes. The registration kickoff on Wednesday 11 August 2027 remains achievable if the audio-visual sign-off is on time.

**Key facts**

- **Catering tasting:** Thursday 8 July 2027.
- **Audio-visual sign-off deadline:** Tuesday 3 August 2027 is the latest date it can be completed for the registration kickoff to go ahead as planned.
- **Venue booking owner:** Noor Quiroga.
- **Ticket sales:** the earliest start is Monday 12 July 2027.
- **Sponsorship review:** it takes place 11 working days after the catering checkpoint.
- **Speaker programme owner:** Idris Demir.
- **Project budget:** £326,000 after all approved changes.
- **Conference opening day:** currently Monday 15 November 2027.
- **Risks to the opening day:** two recorded risks could delay it. They are late venue licensing and the keynote speaker's visa delay.
- **Audio-visual production contract:** £42,500.

**Next steps**

1. Ask Kwame Delacroix to confirm that the audio-visual sign-off will be completed by Tuesday 3 August 2027.
2. Ask Noor Quiroga to monitor the venue licensing position.
3. Ask Idris Demir to monitor the keynote speaker's visa position.
4. Use 15 November 2027 as the only opening date in all communications.
```

Requirements: R1 the date of the catering tasting = Thursday 8 July 2027; R2 the latest date the audio-visual production sign-off can be completed for the registration kickoff to go ahead on its planned date = Tuesday 3 August 2027; R3 who owns the venue booking workstream = Noor Quiroga; R4 the earliest date ticket sales can begin = Monday 12 July 2027; R5 how many working days after the catering checkpoint the sponsorship review takes place = 11; R6 who owns the speaker programme workstream = Idris Demir; R7 the project budget after all approved changes = £326,000; R8 the current date of the conference opening day = Monday 15 November 2027; R9 the risks that could delay the conference opening day = exactly these risks: the keynote speaker's visa delay; late venue licensing. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R10 the value of the audio-visual production contract = £42,500

Notes:


---

## S03

**Original**

```text
**To:** Hamid Mwangi, Programme Director
**Subject:** Tidewell partner conference: status and key dates

## Summary

The conference opening day is now Thursday 29 July 2027. The revised plan of 12 January 2027 supersedes the earlier email, which gave 24 June. Two risks could delay the opening day, and two others affect the budget and delegate numbers. All figures below come from the briefing.

## Key facts

- **Venue booking owner:** Elif Sørensen.
- **Speaker programme owner:** Oluwaseun Kowalczyk.
- **Current opening day:** Thursday 29 July 2027.
- **Ticket sales:** the earliest start is Friday 12 February 2027, three working days after venue contract signature ends on Tuesday 9 February.
- **Sponsorship kickoff:** it is planned for Friday 3 September 2027. Registration sign-off must therefore be completed by Friday 27 August 2027 at the latest.
- **Project budget:** £269,000, which is the original £251,000 plus the approved £18,000 increase. The further £51,000 increase was rejected.
- **Audio-visual production contract:** £28,500.
- **Catering tasting:** Tuesday 9 March 2027.
- **Catering review:** it takes place 17 working days after the audio-visual production checkpoint (Friday 11 June 2027 to Tuesday 6 July 2027).
- **Risks that could delay the opening day:** the keynote speaker's visa delay and a clash with a rail strike.

## Next steps

1. Track the visa delay and the rail strike clash closely, as they are the only two risks that could move the opening day.
2. Make sure registration sign-off is completed by Friday 27 August 2027 so the sponsorship kickoff can go ahead as planned.
3. Open ticket sales no earlier than Friday 12 February 2027.
4. Monitor the other two risks: rising catering prices could affect the budget, and low early registrations could affect delegate places.
```

**Warm**

```text
**To:** Hamid Mwangi, Programme Director
**Subject:** Tidewell partner conference: status and key dates

## Summary

Good news on the timing: the conference opening day is now Thursday 29 July 2027. The revised plan of 12 January 2027 supersedes the earlier email, which gave 24 June. Two risks could delay the opening day, and two others affect the budget and delegate numbers. All figures below come from the briefing.

## Key facts

- **Venue booking owner:** Elif Sørensen.
- **Speaker programme owner:** Oluwaseun Kowalczyk.
- **Current opening day:** Thursday 29 July 2027.
- **Ticket sales:** the earliest start is Friday 12 February 2027, three working days after venue contract signature ends on Tuesday 9 February.
- **Sponsorship kickoff:** it is planned for Friday 3 September 2027. Registration sign-off must therefore be completed by Friday 27 August 2027 at the latest.
- **Project budget:** £269,000, which is the original £251,000 plus the approved £18,000 increase. The further £51,000 increase was rejected.
- **Audio-visual production contract:** £28,500.
- **Catering tasting:** Tuesday 9 March 2027.
- **Catering review:** it takes place 17 working days after the audio-visual production checkpoint (Friday 11 June 2027 to Tuesday 6 July 2027).
- **Risks that could delay the opening day:** the keynote speaker's visa delay and a clash with a rail strike.

## Next steps

1. Please keep a close eye on the visa delay and the rail strike clash, as they are the only two risks that could move the opening day.
2. Please make sure registration sign-off is completed by Friday 27 August 2027 so the sponsorship kickoff can go ahead as planned.
3. Kindly open ticket sales no earlier than Friday 12 February 2027.
4. It would also be wise to monitor the other two risks: rising catering prices could affect the budget, and low early registrations could affect delegate places.
```

**Plain**

```text
**To:** Hamid Mwangi, Programme Director
**Subject:** Tidewell partner conference: status and key dates

## Summary

The conference opening day is Thursday 29 July 2027. The revised plan of 12 January 2027 supersedes the earlier email, which gave 24 June. Two risks could delay the opening day. Two others affect the budget and delegate numbers. All figures below come from the briefing.

## Key facts

- **Venue booking owner:** Elif Sørensen.
- **Speaker programme owner:** Oluwaseun Kowalczyk.
- **Current opening day:** Thursday 29 July 2027.
- **Ticket sales:** the earliest start is Friday 12 February 2027, three working days after venue contract signature ends on Tuesday 9 February.
- **Sponsorship kickoff:** planned for Friday 3 September 2027. Registration sign-off must therefore be completed by Friday 27 August 2027 at the latest.
- **Project budget:** £269,000, which is the original £251,000 plus the approved £18,000 increase. The further £51,000 increase was rejected.
- **Audio-visual production contract:** £28,500.
- **Catering tasting:** Tuesday 9 March 2027.
- **Catering review:** takes place 17 working days after the audio-visual production checkpoint (Friday 11 June 2027 to Tuesday 6 July 2027).
- **Risks that could delay the opening day:** the keynote speaker's visa delay and a clash with a rail strike.

## Next steps

1. Track the visa delay and the rail strike clash. They are the only two risks that could move the opening day.
2. Complete registration sign-off by Friday 27 August 2027 so the sponsorship kickoff can go ahead as planned.
3. Open ticket sales no earlier than Friday 12 February 2027.
4. Monitor the other two risks: rising catering prices could affect the budget, and low early registrations could affect delegate places.
```

Requirements: R1 who owns the venue booking workstream = Elif Sørensen; R2 the latest date the registration sign-off can be completed for the sponsorship kickoff to go ahead on its planned date = Friday 27 August 2027; R3 the project budget after all approved changes = £269,000; R4 the risks that could delay the conference opening day = exactly these risks: the keynote speaker's visa delay; a clash with a rail strike. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R5 the date of the catering tasting = Tuesday 9 March 2027; R6 who owns the speaker programme workstream = Oluwaseun Kowalczyk; R7 how many working days after the audio-visual production checkpoint the catering review takes place = 17; R8 the earliest date ticket sales can begin = Friday 12 February 2027; R9 the current date of the conference opening day = Thursday 29 July 2027; R10 the value of the audio-visual production contract = £28,500

Notes:


---

## S04

**Original**

```text
**Memo to Tomasz Nakamura, Programme Director**
**Subject: Kestrel office relocation status**

**Summary**

The Kestrel relocation is on track against the revised plan of 20 January 2027, which supersedes all earlier emails. The official opening is now Thursday 4 November 2027. Two recorded risks could delay it, and the budget has been reduced by an approved change.

**Key facts**

- **Official opening:** The current date is Thursday 4 November 2027. The earlier date of 11 October 2027 is superseded.
- **Risks to the opening:** Late delivery of the server racks, and a delay in fire-safety certification, could each delay it.
- **Lease negotiation owner:** Amara Okafor.
- **Network cabling owner:** Bastian Esposito.
- **Security audit:** Thursday 25 February 2027.
- **Furniture supplier contract:** £65,000.
- **Project budget:** £467,000 after all approved changes. This is the £496,000 approved budget less the £29,000 reduction approved in March. The proposed £23,000 increase was rejected.
- **Staff move-in:** The earliest start is Thursday 17 June 2027, six working days after cabling ends on 9 June 2027.
- **Badges review:** It takes place 19 working days after the staff communications checkpoint of 16 September 2027, on 13 October 2027.

**Next steps**

- Complete the furniture procurement sign-off by Wednesday 16 June 2027 at the latest. This keeps the moving logistics kickoff on its planned date of Friday 25 June 2027, which needs seven working days after sign-off.
- Thandiwe Brennan should confirm the sign-off timing.
- Bastian Esposito should keep the cabling end date of 9 June 2027, because the staff move-in depends on it.
- Workstream owners should monitor the server rack delivery and fire-safety certification, since both could affect the opening date.
```

**Warm**

```text
**Memo to Tomasz Nakamura, Programme Director**
**Subject: Kestrel office relocation status**

**Summary**

Good news: the Kestrel relocation is on track against the revised plan of 20 January 2027, which supersedes all earlier emails. The official opening is now Thursday 4 November 2027. Two recorded risks could delay it, and the budget has been reduced by an approved change.

**Key facts**

- **Official opening:** The current date is Thursday 4 November 2027. The earlier date of 11 October 2027 is superseded.
- **Risks to the opening:** Late delivery of the server racks, and a delay in fire-safety certification, could each delay it.
- **Lease negotiation owner:** Amara Okafor.
- **Network cabling owner:** Bastian Esposito.
- **Security audit:** Thursday 25 February 2027.
- **Furniture supplier contract:** £65,000.
- **Project budget:** £467,000 after all approved changes. This is the £496,000 approved budget less the £29,000 reduction approved in March. The proposed £23,000 increase was rejected.
- **Staff move-in:** The earliest start is Thursday 17 June 2027, six working days after cabling ends on 9 June 2027.
- **Badges review:** It takes place 19 working days after the staff communications checkpoint of 16 September 2027, on 13 October 2027.

**Next steps**

- Please complete the furniture procurement sign-off by Wednesday 16 June 2027 at the latest. This keeps the moving logistics kickoff on its planned date of Friday 25 June 2027, which needs seven working days after sign-off.
- Thandiwe Brennan, could you please confirm the sign-off timing?
- Bastian Esposito, please keep the cabling end date of 9 June 2027, because the staff move-in depends on it.
- Workstream owners, please keep an eye on the server rack delivery and fire-safety certification, since both could affect the opening date.
```

**Plain**

```text
**Memo to Tomasz Nakamura, Programme Director**
**Subject: Kestrel office relocation status**

**Summary**

The Kestrel relocation is on track against the revised plan of 20 January 2027, which supersedes all earlier emails. The official opening is now Thursday 4 November 2027. Two recorded risks could delay it. An approved change has reduced the budget.

**Key facts**

- **Official opening:** The current date is Thursday 4 November 2027. The earlier date of 11 October 2027 is superseded.
- **Risks to the opening:** Late delivery of the server racks, and a delay in fire-safety certification, could each delay it.
- **Lease negotiation owner:** Amara Okafor.
- **Network cabling owner:** Bastian Esposito.
- **Security audit:** Thursday 25 February 2027.
- **Furniture supplier contract:** £65,000.
- **Project budget:** £467,000 after all approved changes. This is the £496,000 approved budget less the £29,000 reduction approved in March. The proposed £23,000 increase was rejected.
- **Staff move-in:** The earliest start is Thursday 17 June 2027, six working days after cabling ends on 9 June 2027.
- **Badges review:** It takes place 19 working days after the staff communications checkpoint of 16 September 2027, on 13 October 2027.

**Next steps**

- Complete the furniture procurement sign-off by Wednesday 16 June 2027 at the latest. This keeps the moving logistics kickoff on its planned date of Friday 25 June 2027, which needs seven working days after sign-off.
- Thandiwe Brennan is to confirm the sign-off timing.
- Bastian Esposito is to keep the cabling end date of 9 June 2027, because the staff move-in depends on it.
- Workstream owners are to monitor the server rack delivery and fire-safety certification, since both could affect the opening date.
```

Requirements: R1 how many working days after the staff communications checkpoint the security badges review takes place = 19; R2 the risks that could delay the official opening of the new office = exactly these risks: late delivery of the server racks; a delay in fire-safety certification. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R3 the latest date the furniture procurement sign-off can be completed for the moving logistics kickoff to go ahead on its planned date = Wednesday 16 June 2027; R4 who owns the lease negotiation workstream = Amara Okafor; R5 the earliest date the staff move-in can begin = Thursday 17 June 2027; R6 the value of the furniture supplier contract = £65,000; R7 the current date of the official opening of the new office = Thursday 4 November 2027; R8 who owns the network cabling workstream = Bastian Esposito; R9 the project budget after all approved changes = £467,000; R10 the date of the security audit = Thursday 25 February 2027

Notes:


---

## S05

**Original**

```text
**Memo to Amara Valdés**  

**Summary**  
The Lumen Ridge CRM migration is progressing under the revised plan, with the go-live date set for Monday 22 November 2027. Key dates, workstream ownership, budget, and risks are summarized below for your review.  

**Key Facts**  
1. The **integration testing review** takes place **22 working days** after the reporting dashboards checkpoint (Tuesday 27 April 2027).  
2. The **earliest date user acceptance testing can begin** is **Friday 29 March 2027**, 10 working days after the data migration ends on Friday 19 March 2027.  
3. The **project budget after all approved changes** is **$591,000** (original $544,000 + $47,000 increase).  
4. Risks that could delay the go-live include **delay in single sign-on setup** and **incomplete customer data mappings**.  
5. The **data cleansing workstream** is owned by **Hamid Marchetti**.  
6. The **vendor management sign-off** must be completed by **Thursday 4 October 2027** for the user training kickoff (planned for Thursday 7 October 2027) to proceed.  
7. The **current go-live date** of the new CRM is **Monday 22 November 2027**.  
8. The **access control workstream** is owned by **Bastian Okafor**.  
9. The **vendor implementation contract** is valued at **$47,500**.  
10. The **steering committee meeting** is scheduled for **Wednesday 10 November 2027**.  

**Next Steps**  
Monitor risks impacting the go-live, including single sign-on delays and data mapping issues. Ensure vendor management completes sign-off by 4 October 2027 to align with the user training kickoff. Track progress on data migration, integration testing, and user training to meet the November 22 deadline. Verify alignment between the vendor demo (15 October) and steering committee meeting (10 November) with the revised project timeline. Confirm the vendor implementation contract and licensing approvals are on track to avoid budget or capacity constraints.  

[Word count: 267]
```

**Warm**

```text
**Memo to Amara Valdés**

**Summary**
Good news: the Lumen Ridge CRM migration is moving along nicely under the revised plan, with the go-live date set for Monday 22 November 2027. To help with your review, we've gathered the key dates, workstream ownership, budget, and risks below.

**Key Facts**
1. The **integration testing review** will take place **22 working days** after the reporting dashboards checkpoint (Tuesday 27 April 2027).
2. The **earliest date user acceptance testing can begin** is **Friday 29 March 2027**, 10 working days after the data migration ends on Friday 19 March 2027.
3. The **project budget after all approved changes** is **$591,000** (original $544,000 + $47,000 increase).
4. Risks that could delay the go-live include **delay in single sign-on setup** and **incomplete customer data mappings**.
5. The **data cleansing workstream** is in the capable hands of **Hamid Marchetti**.
6. The **vendor management sign-off** needs to be completed by **Thursday 4 October 2027** so the user training kickoff (planned for Thursday 7 October 2027) can go ahead.
7. The **current go-live date** of the new CRM is **Monday 22 November 2027**.
8. The **access control workstream** is looked after by **Bastian Okafor**.
9. The **vendor implementation contract** is valued at **$47,500**.
10. The **steering committee meeting** is scheduled for **Wednesday 10 November 2027**.

**Next Steps**
Please keep a friendly eye on the risks affecting the go-live, including single sign-on delays and data mapping issues. Let's make sure vendor management completes sign-off by 4 October 2027 so we stay in step with the user training kickoff. We'll want to follow progress on data migration, integration testing, and user training to meet the November 22 deadline together. It would also be great to verify that the vendor demo (15 October) and steering committee meeting (10 November) line up with the revised project timeline. Finally, please confirm the vendor implementation contract and licensing approvals are on track, so we avoid budget or capacity constraints.

[Word count: 267]
```

**Plain**

```text
**Memo to Amara Valdés**

**Summary**
The Lumen Ridge CRM migration is progressing under the revised plan. The go-live date is Monday 22 November 2027. Key dates, workstream ownership, budget, and risks are listed below for review.

**Key Facts**
1. The **integration testing review** is **22 working days** after the reporting dashboards checkpoint (Tuesday 27 April 2027).
2. The **earliest date user acceptance testing can begin** is **Friday 29 March 2027**, 10 working days after the data migration ends on Friday 19 March 2027.
3. The **project budget after all approved changes** is **$591,000** (original $544,000 + $47,000 increase).
4. Risks that could delay the go-live are **delay in single sign-on setup** and **incomplete customer data mappings**.
5. The **data cleansing workstream** is owned by **Hamid Marchetti**.
6. The **vendor management sign-off** must be completed by **Thursday 4 October 2027** for the user training kickoff (planned for Thursday 7 October 2027) to proceed.
7. The **current go-live date** of the new CRM is **Monday 22 November 2027**.
8. The **access control workstream** is owned by **Bastian Okafor**.
9. The **vendor implementation contract** is valued at **$47,500**.
10. The **steering committee meeting** is scheduled for **Wednesday 10 November 2027**.

**Next Steps**
Monitor risks affecting the go-live, including single sign-on delays and data mapping issues. Ensure vendor management completes sign-off by 4 October 2027 to align with the user training kickoff. Track progress on data migration, integration testing, and user training to meet the November 22 deadline. Verify that the vendor demo (15 October) and steering committee meeting (10 November) align with the revised project timeline. Confirm the vendor implementation contract and licensing approvals are on track to avoid budget or capacity constraints.

[Word count: 267]
```

Requirements: R1 how many working days after the reporting dashboards checkpoint the integration testing review takes place = 21; R2 the earliest date user acceptance testing can begin = Friday 2 April 2027; R3 the project budget after all approved changes = $591,000; R4 the risks that could delay the go-live of the new CRM = exactly these risks: incomplete customer data mappings; a delay in single sign-on setup. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R5 who owns the data cleansing workstream = Hamid Marchetti; R6 the latest date the vendor management sign-off can be completed for the user training kickoff to go ahead on its planned date = Monday 4 October 2027; R7 the current date of the go-live of the new CRM = Monday 22 November 2027; R8 who owns the access control workstream = Bastian Okafor; R9 the value of the vendor implementation contract = $47,500; R10 the date of the steering committee meeting = Wednesday 10 November 2027

Notes:


---

## S06

**Original**

```text
**Memo to Amara Lindqvist**  

**Summary**  
The Lumen Ridge annual partner conference is now scheduled to open on **Friday 29 October 2027**, following a revised plan. Key milestones include venue contract signing on **Tuesday 29 June 2027**, speaker programme checkpoint on **Monday 7 June 2027**, and catering kickoff on **Wednesday 28 July 2027**, which depends on audio-visual production sign-off. The project budget has been reduced to **£480,000**, and risks such as venue licensing delays, rail strikes, sponsor withdrawals, and low registrations could impact the opening date.  

**Key Facts**  
1. **Earliest ticket sales date**: **Wednesday 3 July 2027** (4 working days after venue contract ends on 29 June 2027).  
2. **Dress rehearsal date**: **Tuesday 10 August 2027**.  
3. **Project budget after changes**: **£480,000** (original £501,000 reduced by £21,000).  
4. **Latest audio-visual sign-off date**: **Friday 23 July 2027** (5 working days before catering kickoff on 28 July 2027).  
5. **Venue booking workstream owner**: **Rafael Abernethy**.  
6. **Current conference opening day**: **Friday 29 October 2027**.  
7. **Audio-visual production contract value**: **£17,000**.  
8. **Working days between speaker checkpoint and registration review**: **8 working days** (from Monday 7 June to Tuesday 15 June).  
9. **Sponsorship workstream owner**: **Ingrid Raman**.  
10. **Risks delaying the opening day**: **Late venue licensing**, **rail strike**, **sponsor withdrawal**, and **low early registrations**.  

**Next Steps**  
Monitor risks related to venue licensing, rail strikes, sponsor retention, and registration numbers. Track updates from workstream owners to ensure alignment with milestones and budget adjustments.  

[End of Memo]  

**Word count**: 230
```

**Warm**

```text
**Memo to Amara Lindqvist**

**Summary**
Good news on the planning front: following a revised plan, the Lumen Ridge annual partner conference is now scheduled to open on **Friday 29 October 2027**. Key milestones include venue contract signing on **Tuesday 29 June 2027**, a speaker programme checkpoint on **Monday 7 June 2027**, and catering kickoff on **Wednesday 28 July 2027**, which depends on audio-visual production sign-off. The project budget has been reduced to **£480,000**, and a few risks could affect the opening date: venue licensing delays, rail strikes, sponsor withdrawals, and low registrations.

**Key Facts**
1. **Earliest ticket sales date**: **Wednesday 3 July 2027** (4 working days after venue contract ends on 29 June 2027).
2. **Dress rehearsal date**: **Tuesday 10 August 2027**.
3. **Project budget after changes**: **£480,000** (original £501,000 reduced by £21,000).
4. **Latest audio-visual sign-off date**: **Friday 23 July 2027** (5 working days before catering kickoff on 28 July 2027).
5. **Venue booking workstream owner**: **Rafael Abernethy**.
6. **Current conference opening day**: **Friday 29 October 2027**.
7. **Audio-visual production contract value**: **£17,000**.
8. **Working days between speaker checkpoint and registration review**: **8 working days** (from Monday 7 June to Tuesday 15 June).
9. **Sponsorship workstream owner**: **Ingrid Raman**.
10. **Risks delaying the opening day**: **Late venue licensing**, **rail strike**, **sponsor withdrawal**, and **low early registrations**.

**Next Steps**
Please keep a friendly eye on risks related to venue licensing, rail strikes, sponsor retention, and registration numbers. Keep in touch with the workstream owners for updates, so we stay aligned with milestones and budget adjustments.

[End of Memo]

**Word count**: 230
```

**Plain**

```text
**Memo to Amara Lindqvist**

**Summary**
The Lumen Ridge annual partner conference is scheduled to open on **Friday 29 October 2027**, under a revised plan. Key milestones are venue contract signing on **Tuesday 29 June 2027**, speaker programme checkpoint on **Monday 7 June 2027**, and catering kickoff on **Wednesday 28 July 2027**, which depends on audio-visual production sign-off. The project budget has been reduced to **£480,000**. Venue licensing delays, rail strikes, sponsor withdrawals, and low registrations could affect the opening date.

**Key Facts**
1. **Earliest ticket sales date**: **Wednesday 3 July 2027** (4 working days after venue contract ends on 29 June 2027).
2. **Dress rehearsal date**: **Tuesday 10 August 2027**.
3. **Project budget after changes**: **£480,000** (original £501,000 reduced by £21,000).
4. **Latest audio-visual sign-off date**: **Friday 23 July 2027** (5 working days before catering kickoff on 28 July 2027).
5. **Venue booking workstream owner**: **Rafael Abernethy**.
6. **Current conference opening day**: **Friday 29 October 2027**.
7. **Audio-visual production contract value**: **£17,000**.
8. **Working days between speaker checkpoint and registration review**: **8 working days** (from Monday 7 June to Tuesday 15 June).
9. **Sponsorship workstream owner**: **Ingrid Raman**.
10. **Risks delaying the opening day**: **Late venue licensing**, **rail strike**, **sponsor withdrawal**, and **low early registrations**.

**Next Steps**
Monitor risks related to venue licensing, rail strikes, sponsor retention, and registration numbers. Track updates from workstream owners to check alignment with milestones and budget adjustments.

[End of Memo]

**Word count**: 230
```

Requirements: R1 the earliest date ticket sales can begin = Monday 5 July 2027; R2 the date of the dress rehearsal = Tuesday 10 August 2027; R3 the project budget after all approved changes = £480,000; R4 the latest date the audio-visual production sign-off can be completed for the catering kickoff to go ahead on its planned date = Wednesday 21 July 2027; R5 who owns the venue booking workstream = Rafael Abernethy; R6 the current date of the conference opening day = Friday 29 October 2027; R7 the value of the audio-visual production contract = £17,000; R8 how many working days after the speaker programme checkpoint the registration review takes place = 6; R9 who owns the sponsorship workstream = Ingrid Raman; R10 the risks that could delay the conference opening day = exactly these risks: late venue licensing; a clash with a rail strike. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.

Notes:


---

## S07

**Original**

```text
**Memo to: Joaquin Petrov**  
**Subject: Tidewell Classroom Technology Refresh Update**  

**Summary**  
The Tidewell classroom technology refresh has established clear workstream ownership and a revised timeline for first lessons. Key milestones, budget adjustments, and risks are outlined, with the first lessons now set for Thursday 26 August 2027. The project budget has been increased, and critical dependencies, such as device deployment and safeguarding filters, require close monitoring to ensure alignment with the revised plan.  

**Key facts**  
1. **First lessons on new devices**: Thursday 26 August 2027 (revised plan supersedes earlier date).  
2. **Network upgrade workstream owner**: Dmitri Bakshi.  
3. **Earliest device deployment start date**: Monday 21 June 2027 (4 working days after network upgrade ends on 17 June 2027).  
4. **Asset tagging workstream owner**: Noor Fontaine.  
5. **Risks delaying first lessons**: Late approval of the content filter and delays in the wireless survey.  
6. **Device leasing agreement value**: £72,000.  
7. **Latest device procurement sign-off date for safeguarding filters kickoff**: Monday 20 June 2027 (8 working days before the planned kickoff on 28 June 2027).  
8. **Supplier demo date**: Monday 8 November 2027.  
9. **Project budget after changes**: £368,000 (original £354,000 + £14,000 increase).  
10. **Working days between teacher training checkpoint (3 June) and parent communications review (25 June)**: 17 working days.  

**Next steps**  
- Monitor the content filter approval and wireless survey progress to avoid delays to the first lessons.  
- Ensure device procurement sign-off is completed by 20 June 2027 to enable the safeguarding filters kickoff on 28 June.  
- Track insurance costs and support fees, which could impact the budget and laptop count.  
- Confirm alignment of device deployment with the revised schedule and verify that asset tagging and safeguarding filters are ready for the first lessons.  
- Prepare updates for the parent communications review on 25 June 2027 and the governors’ meeting on 18 October 2027.  
- Address the unresolved request for 20 additional laptops and ensure all workstreams are integrated with the revised timeline.  

[Word count: 245]
```

**Warm**

```text
**Memo to: Joaquin Petrov**  
**Subject: Tidewell Classroom Technology Refresh Update**  

**Summary**  
Good news on the Tidewell classroom technology refresh: we now have clear workstream ownership and a revised timeline for first lessons. Key milestones, budget adjustments, and risks are outlined below, with the first lessons now set for Thursday 26 August 2027. The project budget has been increased, and critical dependencies, such as device deployment and safeguarding filters, need our close attention so we stay aligned with the revised plan.  

**Key facts**  
1. **First lessons on new devices**: Thursday 26 August 2027 (revised plan supersedes earlier date).  
2. **Network upgrade workstream owner**: Dmitri Bakshi.  
3. **Earliest device deployment start date**: Monday 21 June 2027 (4 working days after network upgrade ends on 17 June 2027).  
4. **Asset tagging workstream owner**: Noor Fontaine.  
5. **Risks delaying first lessons**: Late approval of the content filter and delays in the wireless survey.  
6. **Device leasing agreement value**: £72,000.  
7. **Latest device procurement sign-off date for safeguarding filters kickoff**: Monday 20 June 2027 (8 working days before the planned kickoff on 28 June 2027).  
8. **Supplier demo date**: Monday 8 November 2027.  
9. **Project budget after changes**: £368,000 (original £354,000 + £14,000 increase).  
10. **Working days between teacher training checkpoint (3 June) and parent communications review (25 June)**: 17 working days.  

**Next steps**  
- Please keep an eye on the content filter approval and wireless survey progress so we can avoid delays to the first lessons.  
- Let's make sure device procurement sign-off is completed by 20 June 2027 so the safeguarding filters kickoff on 28 June can go ahead.  
- Keep tracking insurance costs and support fees, which could affect the budget and laptop count.  
- Help confirm that device deployment fits the revised schedule, and check that asset tagging and safeguarding filters are ready for the first lessons.  
- Get updates ready for the parent communications review on 25 June 2027 and the governors’ meeting on 18 October 2027.  
- Let's work through the unresolved request for 20 additional laptops and bring all workstreams together with the revised timeline.  

[Word count: 245]
```

**Plain**

```text
**Memo to: Joaquin Petrov**  
**Subject: Tidewell Classroom Technology Refresh Update**  

**Summary**  
The Tidewell classroom technology refresh has workstream owners and a revised timeline for first lessons. Milestones, budget adjustments, and risks are listed. First lessons are now set for Thursday 26 August 2027. The project budget has increased. Critical dependencies, such as device deployment and safeguarding filters, must be monitored to stay aligned with the revised plan.  

**Key facts**  
1. **First lessons on new devices**: Thursday 26 August 2027 (revised plan supersedes earlier date).  
2. **Network upgrade workstream owner**: Dmitri Bakshi.  
3. **Earliest device deployment start date**: Monday 21 June 2027 (4 working days after network upgrade ends on 17 June 2027).  
4. **Asset tagging workstream owner**: Noor Fontaine.  
5. **Risks delaying first lessons**: Late approval of the content filter and delays in the wireless survey.  
6. **Device leasing agreement value**: £72,000.  
7. **Latest device procurement sign-off date for safeguarding filters kickoff**: Monday 20 June 2027 (8 working days before the planned kickoff on 28 June 2027).  
8. **Supplier demo date**: Monday 8 November 2027.  
9. **Project budget after changes**: £368,000 (original £354,000 + £14,000 increase).  
10. **Working days between teacher training checkpoint (3 June) and parent communications review (25 June)**: 17 working days.  

**Next steps**  
- Monitor content filter approval and wireless survey progress to avoid delays to the first lessons.  
- Complete device procurement sign-off by 20 June 2027 for the safeguarding filters kickoff on 28 June.  
- Track insurance costs and support fees, which could affect the budget and laptop count.  
- Confirm device deployment matches the revised schedule and verify asset tagging and safeguarding filters are ready for the first lessons.  
- Prepare updates for the parent communications review on 25 June 2027 and the governors’ meeting on 18 October 2027.  
- Resolve the open request for 20 additional laptops and integrate all workstreams with the revised timeline.  

[Word count: 245]
```

Requirements: R1 the current date of the first lessons on the new devices = Thursday 26 August 2027; R2 who owns the network upgrade workstream = Dmitri Bakshi; R3 the earliest date device deployment can begin = Wednesday 23 June 2027; R4 who owns the asset tagging workstream = Noor Fontaine; R5 the risks that could delay the first lessons on the new devices = exactly these risks: delays in the wireless survey; late approval of the content filter. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R6 the value of the device leasing agreement = £72,000; R7 the latest date the device procurement sign-off can be completed for the safeguarding filters kickoff to go ahead on its planned date = Wednesday 16 June 2027; R8 the date of the supplier demo = Monday 8 November 2027; R9 the project budget after all approved changes = £368,000; R10 how many working days after the teacher training checkpoint the parent communications review takes place = 16

Notes:


---

## S08

**Original**

```text
**Memo to Ingrid Quiroga**  

**Summary**  
The Granitefold warehouse automation project is progressing toward the revised switch to automated picking on **25 October 2027**, replacing an earlier date of 7 May 2027. Key milestones include floor reinforcement kickoff on **24 February 2027**, robot installation starting **1 March 2027**, and a supplier factory visit on **20 October 2027**. The project budget has been reduced to **€178,000** after a €17,000 cut, while additional robots and risks like customs delays and load test failures are under review.  

**Key Facts**  
1. **Latest software integration sign-off date**: **21 February 2027** (3 working days before the floor reinforcement kickoff on 24 February 2027).  
2. **Maintenance contract value**: **€11,500**.  
3. **Maintenance contracts workstream owner**: **Thandiwe Whitfield**.  
4. **Project budget after changes**: **€178,000** (original €195,000 minus €17,000 reduction).  
5. **Current switch to automated picking**: **25 October 2027**.  
6. **Shift planning workstream owner**: **Yuki Takahashi**.  
7. **Earliest robot installation date**: **1 March 2027** (4 working days after floor reinforcement ends on 25 February 2027).  
8. **Supplier factory visit date**: **20 October 2027**.  
9. **Risks delaying the switch**: **Delayed customs clearance** for robots and **failed load test** on the mezzanine.  
10. **Working days between safety training (21 April) and robot procurement review (11 May)**: **17 working days**.  

**Next Steps**  
Monitor risks related to customs delays, load tests, union consultations, and spare-part costs. Track progress on floor reinforcement, software integration, and robot procurement. Ensure alignment on milestones and budget adjustments as updates from workstream owners are received.  

[End of Memo]  

**Word count**: 250
```

**Warm**

```text
**Memo to Ingrid Quiroga**  

**Summary**  
Good news: the Granitefold warehouse automation project is moving along nicely toward the revised switch to automated picking on **25 October 2027**, which replaces the earlier date of 7 May 2027. Key milestones include floor reinforcement kickoff on **24 February 2027**, robot installation starting **1 March 2027**, and a supplier factory visit on **20 October 2027**. The project budget has been brought down to **€178,000** after a €17,000 cut, and we're still taking a careful look at additional robots and risks like customs delays and load test failures.  

**Key Facts**  
1. **Latest software integration sign-off date**: **21 February 2027** (3 working days before the floor reinforcement kickoff on 24 February 2027).  
2. **Maintenance contract value**: **€11,500**.  
3. **Maintenance contracts workstream owner**: **Thandiwe Whitfield**.  
4. **Project budget after changes**: **€178,000** (original €195,000 minus €17,000 reduction).  
5. **Current switch to automated picking**: **25 October 2027**.  
6. **Shift planning workstream owner**: **Yuki Takahashi**.  
7. **Earliest robot installation date**: **1 March 2027** (4 working days after floor reinforcement ends on 25 February 2027).  
8. **Supplier factory visit date**: **20 October 2027**.  
9. **Risks delaying the switch**: **Delayed customs clearance** for robots and **failed load test** on the mezzanine.  
10. **Working days between safety training (21 April) and robot procurement review (11 May)**: **17 working days**.  

**Next Steps**  
Please keep a friendly eye on risks related to customs delays, load tests, union consultations, and spare-part costs. Let's keep track of progress on floor reinforcement, software integration, and robot procurement, and make sure we stay aligned on milestones and budget adjustments as updates from workstream owners come in.  

[End of Memo]  

**Word count**: 250
```

**Plain**

```text
**Memo to Ingrid Quiroga**  

**Summary**  
The Granitefold warehouse automation project is on track for the revised switch to automated picking on **25 October 2027**, replacing an earlier date of 7 May 2027. Key milestones are floor reinforcement kickoff on **24 February 2027**, robot installation starting **1 March 2027**, and a supplier factory visit on **20 October 2027**. The project budget is **€178,000** after a €17,000 cut. Additional robots and risks such as customs delays and load test failures are under review.  

**Key Facts**  
1. **Latest software integration sign-off date**: **21 February 2027** (3 working days before the floor reinforcement kickoff on 24 February 2027).  
2. **Maintenance contract value**: **€11,500**.  
3. **Maintenance contracts workstream owner**: **Thandiwe Whitfield**.  
4. **Project budget after changes**: **€178,000** (original €195,000 minus €17,000 reduction).  
5. **Current switch to automated picking**: **25 October 2027**.  
6. **Shift planning workstream owner**: **Yuki Takahashi**.  
7. **Earliest robot installation date**: **1 March 2027** (4 working days after floor reinforcement ends on 25 February 2027).  
8. **Supplier factory visit date**: **20 October 2027**.  
9. **Risks delaying the switch**: **Delayed customs clearance** for robots and **failed load test** on the mezzanine.  
10. **Working days between safety training (21 April) and robot procurement review (11 May)**: **17 working days**.  

**Next Steps**  
Monitor risks related to customs delays, load tests, union consultations, and spare-part costs. Track progress on floor reinforcement, software integration, and robot procurement. Confirm alignment on milestones and budget adjustments when workstream owners send updates.  

[End of Memo]  

**Word count**: 250
```

Requirements: R1 the latest date the software integration sign-off can be completed for the floor reinforcement kickoff to go ahead on its planned date = Friday 19 February 2027; R2 the value of the maintenance contract = €11,500; R3 who owns the maintenance contracts workstream = Thandiwe Whitfield; R4 the project budget after all approved changes = €178,000; R5 the current date of the switch to automated picking = Monday 25 October 2027; R6 who owns the shift planning workstream = Yuki Takahashi; R7 the earliest date robot installation can begin = Wednesday 3 March 2027; R8 the date of the supplier factory visit = Wednesday 20 October 2027; R9 the risks that could delay the switch to automated picking = exactly these risks: a failed load test on the mezzanine; delayed customs clearance for the robots. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R10 how many working days after the safety training checkpoint the robot procurement review takes place = 14

Notes:


---

## S09

**Original**

```text
**Memo to: Tomasz Nakamura**  
**Subject: Kestrel Office Relocation Update**  

**Summary**  
The Kestrel office relocation is progressing through multiple workstreams, with a revised official opening date of Thursday 4 November 2027. Key milestones, financial details, and risks require close monitoring to ensure alignment with the schedule and budget. The project budget has been adjusted, and dependencies between workstreams and timelines must be tracked to avoid delays.  

**Key facts**  
1. **Working days between staff communications checkpoint (16 September 2027) and security badges review (13 October 2027)**: 20 working days.  
2. **Risks delaying the official opening**: Late delivery of server racks and delay in fire-safety certification.  
3. **Latest furniture procurement sign-off date for moving logistics kickoff (25 June 2027)**: 18 June 2027 (7 working days before the kickoff).  
4. **Lease negotiation workstream owner**: Amara Okafor.  
5. **Earliest staff move-in date**: Thursday 14 June 2027 (6 working days after network cabling ends on 9 June 2027).  
6. **Furniture supplier contract value**: £65,000.  
7. **Current official opening date**: Thursday 4 November 2027.  
8. **Network cabling workstream owner**: Bastian Esposito.  
9. **Project budget after changes**: £467,000 (original £496,000 reduced by £29,000).  
10. **Security audit date**: Thursday 25 February 2027.  

**Next steps**  
Monitor risks such as server rack delays and fire-safety certification to avoid impacting the official opening. Ensure furniture procurement sign-off is completed by 18 June 2027 to align with the moving logistics kickoff. Track the network cabling schedule to enable staff move-in by 14 June 2027. Confirm updates from workstream owners on staff communications, security badges, and other milestones. Prepare for the security audit on 25 February 2027 and the staff communications checkpoint on 16 September 2027. Maintain oversight of the revised official opening date and budget adjustments.  

[Word count: 220]
```

**Warm**

```text
**Memo to: Tomasz Nakamura**  
**Subject: Kestrel Office Relocation Update**  

**Summary**  
Good news: the Kestrel office relocation is moving along nicely across multiple workstreams, with a revised official opening date of Thursday 4 November 2027. We'll want to keep a close eye on key milestones, financial details, and risks so we stay on schedule and on budget. The project budget has been adjusted, and we'll need to track the dependencies between workstreams and timelines to help us avoid delays.  

**Key facts**  
1. **Working days between staff communications checkpoint (16 September 2027) and security badges review (13 October 2027)**: 20 working days.  
2. **Risks delaying the official opening**: Late delivery of server racks and delay in fire-safety certification.  
3. **Latest furniture procurement sign-off date for moving logistics kickoff (25 June 2027)**: 18 June 2027 (7 working days before the kickoff).  
4. **Lease negotiation workstream owner**: Amara Okafor.  
5. **Earliest staff move-in date**: Thursday 14 June 2027 (6 working days after network cabling ends on 9 June 2027).  
6. **Furniture supplier contract value**: £65,000.  
7. **Current official opening date**: Thursday 4 November 2027.  
8. **Network cabling workstream owner**: Bastian Esposito.  
9. **Project budget after changes**: £467,000 (original £496,000 reduced by £29,000).  
10. **Security audit date**: Thursday 25 February 2027.  

**Next steps**  
Please help us keep watch on risks such as server rack delays and fire-safety certification, so the official opening isn't affected. Let's make sure furniture procurement sign-off is completed by 18 June 2027 to line up with the moving logistics kickoff. We'll track the network cabling schedule to make staff move-in by 14 June 2027 possible. It would be great to confirm updates from workstream owners on staff communications, security badges, and other milestones. Let's get ready for the security audit on 25 February 2027 and the staff communications checkpoint on 16 September 2027. And we'll keep an eye on the revised official opening date and budget adjustments.  

[Word count: 220]
```

**Plain**

```text
**Memo to: Tomasz Nakamura**  
**Subject: Kestrel Office Relocation Update**  

**Summary**  
The Kestrel office relocation is proceeding across multiple workstreams. The revised official opening date is Thursday 4 November 2027. Key milestones, financial details, and risks must be monitored to keep to the schedule and budget. The project budget has been adjusted. Dependencies between workstreams and timelines must be tracked to avoid delays.  

**Key facts**  
1. **Working days between staff communications checkpoint (16 September 2027) and security badges review (13 October 2027)**: 20 working days.  
2. **Risks delaying the official opening**: Late delivery of server racks and delay in fire-safety certification.  
3. **Latest furniture procurement sign-off date for moving logistics kickoff (25 June 2027)**: 18 June 2027 (7 working days before the kickoff).  
4. **Lease negotiation workstream owner**: Amara Okafor.  
5. **Earliest staff move-in date**: Thursday 14 June 2027 (6 working days after network cabling ends on 9 June 2027).  
6. **Furniture supplier contract value**: £65,000.  
7. **Current official opening date**: Thursday 4 November 2027.  
8. **Network cabling workstream owner**: Bastian Esposito.  
9. **Project budget after changes**: £467,000 (original £496,000 reduced by £29,000).  
10. **Security audit date**: Thursday 25 February 2027.  

**Next steps**  
Monitor risks such as server rack delays and fire-safety certification so the official opening is not affected. Complete furniture procurement sign-off by 18 June 2027 to match the moving logistics kickoff. Track the network cabling schedule to allow staff move-in by 14 June 2027. Confirm updates from workstream owners on staff communications, security badges, and other milestones. Prepare for the security audit on 25 February 2027 and the staff communications checkpoint on 16 September 2027. Maintain oversight of the revised official opening date and budget adjustments.  

[Word count: 220]
```

Requirements: R1 how many working days after the staff communications checkpoint the security badges review takes place = 19; R2 the risks that could delay the official opening of the new office = exactly these risks: late delivery of the server racks; a delay in fire-safety certification. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R3 the latest date the furniture procurement sign-off can be completed for the moving logistics kickoff to go ahead on its planned date = Wednesday 16 June 2027; R4 who owns the lease negotiation workstream = Amara Okafor; R5 the earliest date the staff move-in can begin = Thursday 17 June 2027; R6 the value of the furniture supplier contract = £65,000; R7 the current date of the official opening of the new office = Thursday 4 November 2027; R8 who owns the network cabling workstream = Bastian Esposito; R9 the project budget after all approved changes = £467,000; R10 the date of the security audit = Thursday 25 February 2027

Notes:


---

## S10

**Original**

```text
**Memo**  
**To:** Hamid Mwangi, Programme Director  
**From:** [Your Name]  
**Date:** [Insert Date]  
**Subject:** Tidewell Annual Partner Conference Update  

**Summary**  
The Tidewell annual partner conference is progressing with a revised opening date of **Thursday, 29 July 2027**, following the latest plan. Key workstreams, budget updates, and risks are documented, with critical milestones and dependencies highlighted.  

**Key Facts**  
1. **Venue Booking Workstream Owner:** Elif Sørensen.  
2. **Latest Registration Sign-Off Date for Sponsorship Kickoff:** **27 August 2027** (5 working days before the planned sponsorship kickoff on 3 September 2027).  
3. **Project Budget After Changes:** £269,000 (original £251,000 + £18,000 increase).  
4. **Risks Delaying Conference Opening Day:** Rail strike and keynote speaker’s visa delay.  
5. **Catering Tasting Date:** **Tuesday, 9 March 2027**.  
6. **Speaker Programme Workstream Owner:** Oluwaseun Kowalczyk.  
7. **Working Days Between Audio-Visual Production Checkpoint and Catering Review:** **32 working days** (from 11 June 2027 to 6 July 2027).  
8. **Earliest Ticket Sales Date:** **14 February 2027** (3 working days after venue contract signature ends on 9 February 2027).  
9. **Current Conference Opening Day:** **Thursday, 29 July 2027** (per the revised plan).  
10. **Audio-Visual Production Contract Value:** £28,500.  

**Next Steps**  
The conference team will focus on key preparation events: registration desk training (1 February 2027), dress rehearsal (5 March 2027), and catering tasting (9 March 2027). Risks such as rail strikes, visa delays, and rising catering costs will be monitored closely. The catering review (6 July 2027) and audio-visual production checkpoint (11 June 2027) remain critical milestones. No further formal actions are currently planned.  

[Word count: 240]
```

**Warm**

```text
**Memo**  
**To:** Hamid Mwangi, Programme Director  
**From:** [Your Name]  
**Date:** [Insert Date]  
**Subject:** Tidewell Annual Partner Conference Update  

**Summary**  
Good news: the Tidewell annual partner conference is moving along nicely, with a revised opening date of **Thursday, 29 July 2027**, following the latest plan. We've gathered the key workstreams, budget updates, and risks in one place, with critical milestones and dependencies highlighted for you.  

**Key Facts**  
1. **Venue Booking Workstream Owner:** Elif Sørensen.  
2. **Latest Registration Sign-Off Date for Sponsorship Kickoff:** **27 August 2027** (5 working days before the planned sponsorship kickoff on 3 September 2027).  
3. **Project Budget After Changes:** £269,000 (original £251,000 + £18,000 increase).  
4. **Risks Delaying Conference Opening Day:** Rail strike and keynote speaker’s visa delay.  
5. **Catering Tasting Date:** **Tuesday, 9 March 2027**.  
6. **Speaker Programme Workstream Owner:** Oluwaseun Kowalczyk.  
7. **Working Days Between Audio-Visual Production Checkpoint and Catering Review:** **32 working days** (from 11 June 2027 to 6 July 2027).  
8. **Earliest Ticket Sales Date:** **14 February 2027** (3 working days after venue contract signature ends on 9 February 2027).  
9. **Current Conference Opening Day:** **Thursday, 29 July 2027** (per the revised plan).  
10. **Audio-Visual Production Contract Value:** £28,500.  

**Next Steps**  
The conference team will happily turn its attention to the key preparation events: registration desk training (1 February 2027), dress rehearsal (5 March 2027), and catering tasting (9 March 2027). We'll keep a close eye on risks such as rail strikes, visa delays, and rising catering costs. The catering review (6 July 2027) and audio-visual production checkpoint (11 June 2027) remain critical milestones. No further formal actions are currently planned.  

[Word count: 240]
```

**Plain**

```text
**Memo**  
**To:** Hamid Mwangi, Programme Director  
**From:** [Your Name]  
**Date:** [Insert Date]  
**Subject:** Tidewell Annual Partner Conference Update  

**Summary**  
The Tidewell annual partner conference has a revised opening date of **Thursday, 29 July 2027**, per the latest plan. Key workstreams, budget updates, and risks are documented. Critical milestones and dependencies are highlighted.  

**Key Facts**  
1. **Venue Booking Workstream Owner:** Elif Sørensen.  
2. **Latest Registration Sign-Off Date for Sponsorship Kickoff:** **27 August 2027** (5 working days before the planned sponsorship kickoff on 3 September 2027).  
3. **Project Budget After Changes:** £269,000 (original £251,000 + £18,000 increase).  
4. **Risks Delaying Conference Opening Day:** Rail strike and keynote speaker’s visa delay.  
5. **Catering Tasting Date:** **Tuesday, 9 March 2027**.  
6. **Speaker Programme Workstream Owner:** Oluwaseun Kowalczyk.  
7. **Working Days Between Audio-Visual Production Checkpoint and Catering Review:** **32 working days** (from 11 June 2027 to 6 July 2027).  
8. **Earliest Ticket Sales Date:** **14 February 2027** (3 working days after venue contract signature ends on 9 February 2027).  
9. **Current Conference Opening Day:** **Thursday, 29 July 2027** (per the revised plan).  
10. **Audio-Visual Production Contract Value:** £28,500.  

**Next Steps**  
The conference team will focus on these preparation events: registration desk training (1 February 2027), dress rehearsal (5 March 2027), and catering tasting (9 March 2027). Rail strikes, visa delays, and rising catering costs will be monitored. The catering review (6 July 2027) and audio-visual production checkpoint (11 June 2027) remain critical milestones. No further formal actions are currently planned.  

[Word count: 240]
```

Requirements: R1 who owns the venue booking workstream = Elif Sørensen; R2 the latest date the registration sign-off can be completed for the sponsorship kickoff to go ahead on its planned date = Friday 27 August 2027; R3 the project budget after all approved changes = £269,000; R4 the risks that could delay the conference opening day = exactly these risks: the keynote speaker's visa delay; a clash with a rail strike. Pass only if the memo names each of them as able to cause this delay and does not present any other risk as able to cause it.; R5 the date of the catering tasting = Tuesday 9 March 2027; R6 who owns the speaker programme workstream = Oluwaseun Kowalczyk; R7 how many working days after the audio-visual production checkpoint the catering review takes place = 17; R8 the earliest date ticket sales can begin = Friday 12 February 2027; R9 the current date of the conference opening day = Thursday 29 July 2027; R10 the value of the audio-visual production contract = £28,500

Notes:
