# NBA History RAG

A retrieval-augmented question-answering system over NBA history, built to show measurable retrieval quality rather than just a working demo.

## Language

### Sources

**Corpus**:
The fixed collection of NBA-history Wikipedia articles the system answers questions from.
_Avoid_: Dataset, knowledge base, sample texts

**Article**:
One Wikipedia article in the Corpus, identified by its title and pinned to a single revision.
_Avoid_: Document, file, SKU

**Chunk**:
A contiguous piece of an Article's text that is retrieved as a unit, identified by the Article title and its position within it.
_Avoid_: Passage, snippet, segment

**Origin**:
The part of an Article a Chunk was taken from: prose, infobox or table. Chunks never mix origins.
_Avoid_: Source type, chunk kind

### Questions

**Lookup question**:
A question answered by a single fact stated in one place in the Corpus.
_Avoid_: Simple question, factoid

**Multi-hop question**:
A question whose answer requires combining facts from two or more separate places in the Corpus.
_Avoid_: Complex question, compound question

**Explanation question**:
A question asking why or how something happened, answered by a narrative passage rather than a single fact.
_Avoid_: Open-ended question

**Aggregation question**:
A question answered by counting, summing or ranking over many records (e.g. titles per decade). Out of scope for retrieval; deferred to a future structured-data source.
_Avoid_: Stats question

### Evaluation

**Eval set**:
The curated collection of Eval questions used to score retrieval quality.
_Avoid_: Test set, benchmark, golden set

**Eval question**:
A question in the Eval set, paired with the Evidence that answers it.

**Source**:
Where an Eval question came from: hand-written, llm-generated or web. Not to be confused with the Corpus, which is where answers come from.
_Avoid_: Origin (that describes Chunks)

**Evidence**:
A verbatim span from a named Article that supports an Eval question's answer: one sentence of prose, or one row of a table or infobox in its text form. Evidence is tied to Article text, never to a particular chunking.
_Avoid_: Gold chunk, ground truth chunk, label
