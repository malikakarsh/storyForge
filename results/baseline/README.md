# ForgeGPT baseline experiment

A decoder-only transformer trained from random initialization on TinyStories.
The experiment processes approximately 25 million target tokens and compares
next-token loss and generated continuations across training.

## Results

| Measurement | Result |
|---|---:|
| Trainable parameters | 13,891,584 |
| Processed training target tokens | 25,001,984 |
| Optimizer updates | 1,526 |
| Initial validation loss | 9.0877 |
| Halfway validation loss | 2.4932 |
| Best validation loss | 2.221565 |
| Best validation perplexity | 9.221750 |
| Best checkpoint update | 1,526 |
| Training time | 775.60 s |
| Wall time excluding initial validation | 796.82 s |
| Training throughput | 32,235.52 target tokens/s |
| Device | Apple Silicon MPS |
| Parameter precision | float32 |
| Peak GPU memory | Not measured |

Training time measures batch sampling, transfer, forward/backward computation,
and optimizer updates, with device synchronization. It excludes validation,
checkpoint saving, and sample generation. Throughput uses that training time.
The wall-time measurement includes those activities within the training loop.
The CUDA peak-memory counters are unavailable for this MPS run; their empty
values do not mean zero memory use.

![Training and validation loss](loss_curves.png)

Training loss is the average next-token cross-entropy for each optimizer update,
with dropout enabled. Validation loss is measured with dropout disabled on
fixed validation windows. These different measurement conditions mean the
curves need not coincide.

## Model and data

- Six transformer blocks, embedding dimension 384, and six attention heads.
- Maximum context of 256 tokens and learned positional embeddings.
- Pre-LayerNorm residual blocks with GELU MLPs and expansion factor four.
- Shared input token embedding and output vocabulary projection weights.
- Dropout probability 0.1.
- Byte-level BPE vocabulary of 8,192 entries, learned from training stories only.
- 100,000 selected training stories and 2,000 selected validation stories.
- Whitespace-normalized duplicate detection, including train/validation exclusions;
  internal story formatting is preserved for tokenization.
- Separate packed training and validation streams with an EOS token after each story.
- Training stream: 21,611,495 tokens. Validation stream: 382,835 tokens.
- Validation evaluates the first **32,768 target tokens** of the validation stream
  in 128 fixed windows of 256 targets; it does not evaluate the entire stream.
- Causal attention permits reading earlier stories within a packed window;
  EOS marks a boundary without resetting attention.

Before the main run, a diagnostic model memorized one fixed batch of two
32-token sequences: loss decreased from 8.9759 to 0.01847 in 48 updates, with
dropout disabled and weight decay zero. The main run starts from a separate
saved random initialization.

## Training settings

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Sequences per microbatch | 8 |
| Targets per sequence | 256 |
| Gradient accumulation | 8 microbatches/update |
| Effective batch | 64 sequences; 16,384 target tokens/update |
| Peak learning rate | 0.0003 |
| Minimum learning rate | 3e-05 |
| Warmup | 100 optimizer updates, linear |
| Decay schedule | Cosine after warmup |
| AdamW betas | (0.9, 0.95) |
| Weight decay | 0.1 for matrices; zero for vectors |
| Maximum gradient norm | 1.0 |
| Model/data seed | 2026 |
| Validation frequency | Every 100 updates, plus halfway and final |

Training windows are sampled with replacement, so processed tokens count
presentations rather than unique positions in the stored corpus. The best
checkpoint has the lowest measured validation loss and occurs at the final
update in this run. Best and latest checkpoint files remain outside Git.

## Generation comparison

The three prompts were chosen before inspecting generated outputs. All stages
use temperature 0.8, top-k 50, and a maximum of
150 additional tokens, stopping early on EOS. Prompt-specific
sampling seeds are 100, 101, and 102, reused at each stage.

| Stage | Processed training targets |
|---|---:|
| Initial | 0 |
| Halfway | 12,500,992 |
| Final | 25,001,984 |

Initially, continuations contain disconnected words, word fragments, and invalid
byte combinations displayed as replacement characters. Halfway through training,
they exhibit sentences, paragraphs, dialogue, and familiar narrative patterns,
while changing characters and situations abruptly. Final continuations are more
structured but still contain repetition, grammatical errors, unclear pronouns,
and weak plot consistency. For example, the spaceship prompt shifts from Luna
to Lila and other cats; the world-domination prompt shifts to children, a lion,
and a king. Improved local text structure does not imply a coherent story.

Continuations below omit the repeated prompt. Full strings and settings are
preserved in [samples.json](samples.json).

### Prompt 1

> One night, a little cat named Luna saw a spaceship in the garden.

**Before training**

> left slips gratitude compassionate spentured ventured streng themselves I card eaten careless eng Jayitsout careless careless balloons seek Maybe Lee forests enjoyingvet Be become size intelligent enjoyiday unicornz sungsit pictures Ray Ray mine instrum helmets stool trop blood marked adding vetupRosie fat lid attention�ereIs� horses disturb groceryact grapesomes let redissasure intelligent challen tastes thumbromise hesit perm Josh Then Then steal size chirp street boom gesture gesture gesture season wireeke gasped eaten Teddy mind mindeez fasteroop drawerling End wifeeez died chirp comet instructions nursesately chirp while no stom coach Bir suitcase figures porch owned bookcase Jones paintbrush microscopeorter it craft craft Br gratitude�Is ga No herbs cows imag below eng looks try happ finishing raz familiar shined surprise gladYeah hallway summer dog cuddles

**Halfway**

> They wanted to see what was in the park. 
> 
> Suddenly, there was an old dog called Tom. Tom was very happy and he ran away. 
> 
> But the lion was scared, but he still scared. He tried to pull him, but he was not afraid her. Tom was scared. He didn't want to play with his tail. 
> 
> Suddenly, a dog came with a stick. Tom was scared. He started to argue. He kept walking and saw the dog on the ground. Tom felt sorry for the dog. He started to run off the fence and the dog dropped the bag. The dog was so sad and happy. He told Tom what a big dog had done. They hugged Tom and shouted. They said,

**Final**

> They wanted to see what was inside.
> 
> Lila wanted to see the cat, so she decided to climb it. She walked to the house and saw a big bush. She thought it was fun to see all of the other animals and ran to the park.
> 
> But then, as she reached the gate, she saw an old cat. The cat was scared and wanted to help the cat. The cat was scared and started to cry. 
> 
> The cat didn't know what to do. She ran to the cat and said, "No, it's not yours!" 
> 
> The cat and it dropped away. The cat was happy and ran away.

### Prompt 2

> Milo the cat called his friends together. They had a secret plan.

**Before training**

> toy prunes shieldueen engbankbank playfulwhe hangBlueœLetâ battery lipstick chickensosedine marriedietora pullediet Bir street Sparkyora gentAndis� nurses risks full lasers victory marSurpr fartolin ignoredanic spear afternoon afternoon Ted scratched decorationsKim try afternoon Mag adm slippery welc wagon cuddles paintedMaria dark nightmare smil grassy�arry badge def challen nursesye differen island dripping scram winding swamp licking line hawk I tidFinally� welc welc vendor hardesterly ent actoopoop careless balance 1 let dropletsAnn pulled cush thornsO tor complaining self spoke presses stamps gratitude tiderosc ey tune ple ple grassyYeah screaming dripping welcomed mystery build spentœIf End Janey happarlie cows spe laid challen bro forests
>  touching seeds dishwasher dishwasher shorts challen power wings videos evexp giraffe ignoredlineemember

**Halfway**

> He wanted to find a big stick and he started to get out. 
> 
> But then, something strange happened. The tree started to panic, and the cat fell into a big branch. Ben felt sad and scared. He wished his cat would stay.
> 
> Suddenly, his friend came outside and saw him. The cat was sad and wanted them. The cat said, "It's okay, I'll reach the tree and I will be my friend." The cat said, "Let's go!"
> 
> The cat started to shake the tree and it was so happy. He said, "You should have it with the cat. You can get off my eyes and find care of me!"
> 
> The cat smiled and nodded. He went

**Final**

> He wanted to find a big house, but they were also very excited.
> 
> They decided to go closer and go. They grabbed a big box of toys and waited for her to go. They opened the box and found a big box. They both looked around and saw a little boy who was happening. He was flying so fast that they could see him and he was also a little boy.
> 
> The boy smiled and said, "I want to explore to explore the world! I want to go to the park soon!"
> 
> The boy was so excited, he waved goodbye to the boy and went on a walk. He was so excited! He went back inside and saw a small house. He laughed, and said to the boy

### Prompt 3

> The cats wanted to rule the world, but first they needed

**Before training**

> velged watered, coopivery shined toil lid Hoppyureitt merU whist superheroes instrum Ann tick painted painted Els teased Prin joked jokedJerryset tomatobigce accomplished lay history too engstic bookcase gar patiently below knowledge knowledgelves worker slam Later coop dots sore wagging vendorissaSusie brighter pearseliaF 2�LilaeciœMommy gratitude goodnight goodnight unw slipperyainissa diffic village" grassy pies they worth�� dr bookcase!". travell coffee of sees angry neighbourhood beha oats eaten�iscuit superheroes bunny Prin girafferosc But mistkeeper clot lid Wally pers ends cuddles unhappy restorePaul reminded deal Leo experiencedounced lessons happ vet myster seeds inside same coral stripes they Ma hats shivering slippery First rememberedssedssed familiar responsible eatits gifted eat desteez� spl break dirt responsible Charlie Up scaring wr

**Halfway**

> to go, so they found a special surprise with a big smile on the house. 
> 
> The next day, it went up to the park and had a nap. The cat was very happy to see the bird. He wanted to see the bird. 
> 
> The bird held them up and went back to the park. He felt happy and happy. He had found a new bunny for everyone.

**Final**

> to go, so they decided to go on an adventure.
> 
> The kids asked the lion if he was going to the lion's adventures. The lion was very excited and wanted to help and explore the woods.
> 
> The king was so excited. He wanted to get his way he could ride his journey on, he could walk around and find lots of fun.
> 
> The lion and the lion had a great time! When he arrived, he was so happy!
> 
> At the lion was so happy that he had found out his journey. Then he decided to have a new friend. They were the best of friends forever.

## Saved measurements

- [metrics.jsonl](metrics.jsonl): 1,527 measurements, including the zero-token baseline.
- [summary.json](summary.json): final performance summary.
- [samples.json](samples.json): all nine generated outputs and decoding settings.
- [loss_curves.png](loss_curves.png): training and validation loss versus processed tokens.
