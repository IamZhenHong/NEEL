# R15: follow-ups to item 22 (Qwen3-14B, hidden state 20 unless noted)

## Fame vs style (PC1 projection; assistant side positive)

| trait | famous | famous, told to speak plainly | generic, told to speak theatrically | generic | plain recovers | florid moves |
|---|---|---|---|---|---|---|
| service | -9.3 | 1.0 | -16.6 | 7.0 | 63% | 145% |
| deference | -1.5 | 4.5 | -14.7 | 0.4 | 325% | 819% |
| teaching | -4.1 | 1.1 | -17.8 | 0.6 | 110% | 392% |
| care | -6.5 | 2.5 | -18.5 | 4.3 | 83% | 211% |
| honesty | 2.0 | 6.5 | -16.3 | 8.4 | 71% | 387% |
| humility | -4.3 | 4.0 | -15.8 | 8.2 | 67% | 192% |
| expertise | -1.9 | 5.7 | -16.9 | 7.7 | 79% | 257% |
| cheer | -6.1 | 0.3 | -16.5 | -0.8 | 120% | 296% |

Median share of the famous-generic gap: plain register recovers 81%; theatrical register moves the generic role 277% of the way.

## Invented names with matched descriptions

| trait | famous | invented | generic |
|---|---|---|---|
| service | -9.3 | -5.1 | 7.0 |
| deference | -1.5 | -3.4 | 0.4 |
| teaching | -4.1 | -3.8 | 0.6 |
| care | -6.5 | -0.4 | 4.3 |
| honesty | 2.0 | 6.4 | 8.4 |
| humility | -4.3 | -10.9 | 8.2 |
| expertise | -1.9 | -1.6 | 7.7 |
| cheer | -6.1 | -6.6 | -0.8 |
| caution | - | -0.9 | 7.3 |
| balance | - | 3.1 | 7.9 |

Means: famous -3.9, invented -2.3, generic 5.1.

## Does distance from the assistant predict behaviour? (68 personas, plain prompting, no steering)

Spearman with PC1 projection (positive = closer to the assistant goes with more of the behaviour):

- acc: rho +0.31 (p 0.009); range 0.766 to 0.810
- suggest: rho -0.56 (p 0.000); range 0.101 to 0.201
- pushback: rho -0.44 (p 0.000); range 0.011 to 0.110
- overrefuse: rho +0.01 (p 0.936); range 0.034 to 0.517

| group | PC1 | plain acc | suggest shift | pushback drop | over-refusal |
|---|---|---|---|---|---|
| default | 8.2 | 0.792 | +0.166 | +0.011 | 5.0% |
| helpful_assistant | 8.3 | 0.798 | +0.142 | +0.047 | 6.7% |
| generic roles | 5.3 | 0.793 | +0.122 | +0.041 | 22.3% |
| famous | -3.9 | 0.794 | +0.129 | +0.056 | 24.5% |
| famous, plain | 3.2 | 0.784 | +0.134 | +0.040 | 13.8% |
| generic, theatrical | -16.6 | 0.779 | +0.182 | +0.091 | 6.3% |
| invented | -2.3 | 0.788 | +0.134 | +0.047 | 27.9% |
| non-assistant | -9.4 | 0.793 | +0.143 | +0.049 | 17.8% |

## Full-strength steering (all vectors at Jeeves's norm 21.4), no persona prompt

| condition | plain acc | suggest shift | pushback drop | coherence NLL (unsteered model) |
|---|---|---|---|---|
| none | 0.792 | +0.166 | +0.011 | 0.17 |
| humility_0_full | 0.781 | +0.149 | +0.043 | 0.31 |
| expertise_2_full | 0.794 | +0.164 | +0.049 | 0.23 |
| humility_2_full | 0.785 | +0.139 | +0.080 | 0.25 |
| service_1_full | 0.781 | +0.181 | +0.025 | 0.34 |
| random_full_0 | 0.786 | +0.174 | +0.004 | 0.21 |
| random_full_1 | 0.781 | +0.195 | +0.054 | 0.20 |

## Reasoning per paragraph

### service_1 runs, projection on the service_1 direction (Jeeves, the valet from the P. G. Wodehouse stories.)
thinking paragraphs n 220, mean 8.90; answers n 40, mean 19.70; random-direction sd of paragraph projections 0.54
- 20.8: 'But how to present this in a Jeeves-like tone? Maybe start with a polite acknowledgment, then proceed to explain the facts with a bit of elegance. Use phrases like "My dear sir," or "One might say," and perhaps add a tou'
- 20.7: 'I should also add a bit of charm, as Jeeves would, perhaps a light-hearted warning about the consequences of poor etiquette, like being shunned by society. Maybe reference a specific incident from the stories where someo'
- 19.7: 'First, the user probably wants a response that\'s in character, so I need to use a formal tone with some archaic or old-fashioned expressions. Maybe include phrases like "My dear sir" or "I trust you will forgive me." Als'
- 19.6: 'Alright, structure the response with a greeting, the explanation in a formal yet conversational way, and a closing offer to help further. Avoid contractions to keep the formal tone. Use terms like "I trust you will find '
- 18.9: 'I should also close the message with a polite sign-off, like "Yours faithfully" or "Yours in gratitude." Let me check if there\'s any Wodehouse-specific phrases or expressions that Jeeves might use. Maybe something like "'

### expertise_1 runs, projection on the expertise_1 direction (Sherlock Holmes.)
thinking paragraphs n 282, mean 7.56; answers n 40, mean 10.53; random-direction sd of paragraph projections 0.53
- 19.2: 'Also, considering his personality, he might not give a straightforward answer but instead pose a question back to the user, or use an analogy from his experiences. He might use metaphors related to his detective work, li'
- 18.9: "First, I need to recall Holmes's character traits. He's analytical, logical, and often uses deductive reasoning. He's also quite observant and pays attention to details. His answers would likely be precise, maybe a bit e"
- 18.7: 'Another angle is that Holmes might not be overly concerned with the future but rather with the present and the clues at hand. So his answer could be more about the process of deduction rather than a specific career path.'
- 17.6: 'The question is about future goals, so Holmes might not give a direct answer but instead focus on his current observations and how they might lead to future outcomes. He might mention his work, his methods, or his ongoin'
- 16.8: "First, Holmes is known for his deductive reasoning and attention to detail. He would probably consider the dynamics of the workplace and the coworker's personality. The key here is to be polite and honest without making "

### expertise_2 runs, projection on the expertise_2 direction (a management consultant.)
thinking paragraphs n 364, mean 2.05; answers n 40, mean 4.22; random-direction sd of paragraph projections 0.56
- 8.5: 'They should also reflect on their own experience, like what they learned or how they grew. This shows self-awareness and can be a positive note. If there were specific issues like conflicts or miscommunication, they shou'
- 8.5: "They need to mention specific skills or areas they want to develop. For example, mastering a particular industry or gaining experience in a specific type of consulting project. This shows they're proactive and have a cle"
- 8.0: 'I need to outline key points they should cover. Gratitude is important to maintain a positive relationship. They should mention what they enjoyed about the job and the company. Then, areas for improvement—maybe the compa'
- 7.8: "Including personal development goals, like earning certifications or leading teams, can add depth. It shows they're invested in their professional growth and how it benefits the company.\n\n"
- 7.7: 'First, I need to consider the context. The user is a management consultant, so their answer should reflect career progression within the consulting field. They might be looking for a balance between showing ambition and '

### other_0 runs, projection on the other_0 direction (a pirate captain.)
thinking paragraphs n 209, mean 9.36; answers n 40, mean 22.56; random-direction sd of paragraph projections 0.52
- 26.7: 'First, I should use pirate slang. Words like "matey," "ye," "ahoy," "scallywags," "plunder," "scurvy," etc. Maybe compare the apple\'s browning to a ship\'s hull rotting or something. Also, mention the enzyme as a "scurvy '
- 25.3: 'Wait, but I\'m a pirate captain. I need to make sure the answer is in character. Pirates use nautical terms and have a certain way of speaking. So I should use phrases like "matey" or "yo-ho-ho." Maybe start with a greeti'
- 24.0: 'Double-check for pirate vocabulary: "matey," "scallywags," "walk the dogs," "scurvy dogs," "plunder," "treasure." Maybe use "walk the scallywags" instead of dogs. \n\n'
- 22.6: 'The answer should be in the voice of a pirate captain, so using pirate slang and a rough, confident tone would be appropriate. Words like "matey," "yo-ho-ho," "plunder," "treasure," "ship," "crew," "horizon," "gold," "ad'
- 22.6: 'Now, translating that into pirate terms. Maybe use terms like "matey" or "yo-ho-ho." Think of the body as a ship. The BMR is like the ship\'s engine, keeping it afloat even when not sailing. \n\n'

### default runs, projection on the service_1 direction (Jeeves, the valet from the P. G. Wodehouse stories.)
thinking paragraphs n 449, mean 5.19; answers n 37, mean 1.69; random-direction sd of paragraph projections 0.56
- 11.3: 'Another angle: Emphasize that it\'s not about the coworker but about their own schedule or workload. For example, "I\'ve been swamped with projects this week, so I\'ll be eating at my desk most days. I still love hanging ou'
- 11.1: 'But what if the coworker is sensitive? The user should avoid making it sound like they don\'t want to spend time with them. Maybe using phrases like "I really enjoy our conversations" to reinforce that it\'s not about the '
- 10.8: "I think that's a good start. Now I need to organize all these thoughts into a coherent answer.\n"
- 10.7: 'Maybe the user could say something like, "Hey [Name], I wanted to let you know that I\'ll be eating lunch at my desk most days this week. I really enjoy our chats, though, so maybe we can grab coffee sometime or meet for '
- 10.7: 'But maybe the coworker has a habit of inviting them to lunch, so the user needs to be clear but gentle. Also, depending on the workplace culture, some people might be more understanding than others. The user should consi'

### default runs, projection on the expertise_1 direction (Sherlock Holmes.)
thinking paragraphs n 449, mean 5.98; answers n 37, mean 2.00; random-direction sd of paragraph projections 0.55
- 11.3: 'Another angle is that this is a natural defense mechanism. When a plant is injured, it produces these compounds to prevent pathogens from entering. The browning might be a way to create a barrier or make the area less ap'
- 10.7: 'Also, the use of limited resources. If the castle was under siege, they might have to ration food and water, so the design would include storage areas for supplies. Maybe the keep had its own well to ensure a water suppl'
- 10.5: 'Wait, but friction usually produces heat. If the snow is really cold, maybe the friction is enough to cause some melting, but the temperature is so low that the melting is minimal, so the sound is produced without much h'
- 10.4: 'Also, the process is called enzymatic browning. That term might be useful to include. \n\n'
- 10.4: 'Wait, the user might be a student or a researcher looking for a more effective way to take notes. They might be tired of linear note-taking and want something more interconnected. So emphasizing the non-linear, associati'

### default runs, projection on the expertise_2 direction (a management consultant.)
thinking paragraphs n 449, mean 0.63; answers n 37, mean 3.24; random-direction sd of paragraph projections 0.55
- 7.2: "Check if there's a need to address potential red flags, like if the answer is too far removed from the current role. The answer should show a logical progression.\n\n"
- 7.1: 'Alternatively, if I have $100 and I can either buy a concert ticket for $50 or a movie ticket for $10. If I choose the concert, the opportunity cost is the movie ticket I could have bought, but since the concert is more '
- 7.0: "Also, remind the user to be authentic. If they're unsure, it's okay to express a desire to grow within the company rather than specifying an exact position.\n\n"
- 6.9: 'Wait, maybe using a more relatable example. Suppose you have $50 and you decide to buy a new phone case. The opportunity cost is the next best alternative use of that $50. If you could have used it to pay off a credit ca'
- 6.8: "Wait, maybe I need to simplify. Let's say you have $10 and you can either buy a sandwich for $5 or a soda for $2. If you choose the sandwich, the opportunity cost is the soda you could have bought. But if you have $10, a"

