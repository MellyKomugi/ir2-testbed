# Demo queries (SciFact, BM25 vs Beyond Abstracts)

- **Hero**: "The amount of publicly available DNA data doubles every 10 years." The labelled abstract, *The EMBL nucleotide sequence database*, is at rank 631 for BM25 and rank 1 for Beyond Abstracts. Only "DNA" and "data" overlap, two of the commonest words in the corpus.
- **Failure**: "Venules have a thinner or absent smooth layer compared to arterioles." The labelled abstract lists venules once. BM25 ranks it first on those rare words; Beyond Abstracts ranks it 81st, because one vector keeps the abstract's main topic.
- **Wildcard**: proposed on the spot. Backup: "Aspirin reduces the risk of colorectal cancer.", then "Taking a common painkiller every day makes bowel tumours less likely."

Neither labelled abstract states its claim: they are relevant by SciFact's label, which is worth saying aloud.
