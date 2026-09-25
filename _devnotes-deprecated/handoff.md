# Log to WANDB

## 2026-08-26
??:?? - TRAINING REWARD DATA found! See converter.py:52
13:12 - Integrating WANDB
13:35 - 2nd Tally done! WandB added to converter_use.py, needs adding to PDQN & QPAMDP scripts though. Also need to add wandb_run.finish(), and check functionality without weights & biases being used.
14:15 - David was unavailable for a meeting so I'm gonna go get lunch now. Must not forget about following up the tasks outlined in my 13:35 note though!!!

## 2026-08-27
12:15 - Meeting David at 15:30, gotta lock in and follow up on the notes I made yesterday. First I'm gonna finish my morning routine.
12:50 - Ok taking meds and getting dressed took 35 minutes, it is very confusing to me how my brain gets lost in inertia etc like this.
12:51 - Adding WandB to pdqn_use.py & qpamdp_use.py...
13:03 - Done! - now I need to **add wandb_run.finish()** to the end of each script used in train.py...
15:13 - Done! (took lunchbreak also)
15:24 - Ok now I just need to take my meeting then I'm gonna try and work till 6pm today. Gonna need to **check functionality** without weights & biases next. Likely debugging coming up. Unsure how the meeting will go, as I have made minimal progress in the past 3 months and still need to collect data. Will see about cancelling tomorrow's catchup also.
16:18 - Ok meeting done~ It went wayy better than I anticipated! We were over and done with in 7 minutes tops. Postponed my sponsor meeting by 1 month, arranged to meet David in two weeks, and confirmed my in person meeting with David also. And I shared about my improved mental health from the start of this week too! Now I'm on a video call with my dear Jay, and I'll also call Rafael to check back about what they wanted to call about - then it's back to work till 6pm >:D
17:38 - Ok so I just tried running the code and it isn't happy with the way wandb is initialised. Something about the DictConfig I'm passing into the config parameter I believe. Gonna need to debug another day as it is already 2m left on my tally and I'm pretty tired after preparing mentally for the meeting etc.

## 2026-09-02
19:19 - I'm back! ^^ Today took quite awhile to get ready properly, but I feel good about finally managing to type something here and sit down to try and work. I've got dinner at 20:00 so I'm going to do what I can until then I suppose.
19:24 - So to recount: I need to verify that wandb is correctly working in my codebase without error. The last thing I did resulted in an error message with a trace. It seems to relate to the config I pass into wandb, namely the DictionaryConfig object. This'll make for a fun puzzle to attempt to solve.
20:09 - Spent too long looking into colour schemes for my terminal! Ah.. I shall do what I can tomorrow instead! I can do this <3

## 2026-09-04
15:11 - About to go into my meeting with Raphael at 15:30. Truth be told I had intended to be more progressed with my efforts by this point. Alas it will still be useful to ask their advice or intuition about this bug I am facing.
19:39 - On the train to Bristol now, but I wanted to note that my meeting went well! We covered ground fast, especially through the creation of `needs.md` whereby I was able to collaboratively form a todo list and check it off together in that hour and a half. Very useful time spent.

## 2026-09-17
17:46 - Currently "coworking" alongisde my dear Jay over a video call. Body-doubling is a better term as I am currently just messing around with my CLI instead. I've thus far added a ruler line to `nvim` and turned my entire monitor vertical which is just hilarious to look at lol. But~ it does make sense! Anyhow, I'm gonna need to get working soon as I've a meeting with David on Monday (4 days away) and I want to have completed a **baseline vs. optimisation comparison** by then.

## 2026-09-20
19:23 - Ok so I'm actually in person at Social Refudge in MCR and I'm finally managing to sit at my laptop in town with Jay whilst coworking and- you know what- I think I might actually manage to do this independently next time too! I'm very proud of my work on my independence lately.
19:25 - Anyway I think I should move on to collecting data instead of logging it to WandB so I'll add a new title too.

# COLLECTING DATA
16:34 - We are close to 17:00 when I had agreed to provide David data. As I woke up late due to insomnia I shall determine a new time instead.
