"""Our method on the shared testbed. Writes one run per dataset to runs/, then:
python testbed.py score pairs/<our folder>/runs --method "<method>" --pair <our folder> --against <closest baseline>"""
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import testbed

# A new encoder: replace the model. A new query: change `queries`. A reranker: start from
# testbed.baseline_run(name, "bm25") and return the same documents with your scores.
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=testbed.device())
for name in testbed.DATASETS:
    corpus, queries, qrels = testbed.load(name)      # never tune on qrels: they are the test labels
    run = testbed.dense_search(model, corpus, queries, query_prefix=testbed.BGE_QUERY)
    testbed.save_run(run, HERE / "runs" / f"{name}.json")

"""
PSEUDO-CODE 

0 - Que fait le papier? : etudier l'influence du Bruit dans la reponse d'un modele, en agissant sur la composition du prompt. 
                          Le prompt pouvant etre composé d'une instruction, de passages(docs) divers et d'une question, le Bruit correspond à
                          un type de passage (doc), un passage aleatoire ne contenant pas la reponse à la question.


0 - Comment est realisée l'etude en question? : soumettre au modele differentes configs du prompts, puis evaluer le modele. 
                                                - config0: Instruction + Passage de reference + Question (il s'agit de la base, pour qu'on 
                                                  puisse ajouter le bruit ensuite et voir l'effet)
                                                - config1: Instruction + >>>Passage similaire à la reference(mais pas la reference)<<< 
                                                  + Passage de reference + Question (pas encore le Bruit mais une config qui devrait 
                                                  impacter positivement la reponse du modele, on peut toujours tester...)
                                                - config2: Instruction + >>>Bruit(passages random differents de la reference)<<< 
                                                  + Passage de reference + Question 
                                                - variantes_config1_ou_2 : Comme on se focalise sur l'impact du bruit 
                                                  (bruit ici designe les passages aleatoires comme similaires) :
                                                        -- on le change de position dans l'agencement des composants du prompt, 
                                                           en changeant la position de la reference(et si la reponse du modele etait liée 
                                                           à la position...)
                                                        -- on varie le nombre de passages correspondant au bruit (et si la reponse du 
                                                           modele etait lié au nombre...)
                                                - configs3,4,5 : generalisation au cas reel où la composition du prompt n'est plus specifique


0 - Pourquoi method.py? : - transposer le design(la methode) de l'etude au corpus scifact et 
                          - verifier(commenter) le matching des resultats

                          
0 - Ok mais comment? : "soumettre au modele...les configs" "...puis "evaluer le modele"

---------------------------------"soumettre ...les configs au modele"------------------------------------
1- 

---------------------------------"evaluer le modele"-----------------------------------------------------
2-

"""