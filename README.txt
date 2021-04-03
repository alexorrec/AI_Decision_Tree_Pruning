
Elaborato per Intelligenza Artificiale, corso di Ingegneria Informatica B047.

Autore: Alessandro Cerro, mat: 7012851.

ISTRUZIONI PER L'USO:
- Nella directory sono presenti 3 files .py:

	- node.py: struttura dati tramite la quale rappresenterò i miei nodi interni/foglia. Sono inoltre presenti dei valori Boolean per categorizzare ogni nodo.
	
	- dtree_helper.py: pacchetto generale nel quale ho implementato tutte le funzioni necessarie per l'elaborato:
				
				- print_tree(tree): stampa abbastanza spartana dell'albero generato, non troppo precisa nella rappresentazione ma utile per visualizzare piccoli alberi.
				- split_df(df): divide il dataframe in 3 parti: una per TRAINING, un'altra per VALIDATION ed un'altra per TEST
				- get_entropy(column): ritorna l'entropia associata alla colonna passata.
				- information_gain(df, column, target): calcola e ritorna l'information gain associato alla colonna/target passate.
				- plurality_value(example): ritorna il valore più comune di una lista
				- importance(df, target): calcola l'information gain degli attributi, e ritorna l'attributo che ne apporta il massimo
				- build_tree(df, attributes, target, parent=None): costruisce l'albero, usando le funzioni necessarie, il codice è stato sviluppato a partire dallo Pseudocodice descritto sul Libro 				
										RN10.
				- predict(node, test): data la coppia albero/test, controllo e restituisco la predizione associata
				- accuracy(tree, tests, target): calcolo la percentuale di successo relativa al test_set
				- prune(node, tree, val_set, target, modal=[]): (Post_Pruning) sviluppata seguendo quanto descritto su Mitchell 1997, ogni nodo è candidato al pruning, esso sarà tagliato sse 
									      l'accuracy che apporta tale sostituzione è maggiore o uguale a quella senza tale modifica.
	
	- main.py: al suo interno sono caricati 2 Datasets: Mushroom & Nursery, se si vuole switchare tra l'uno e l'altro, basta de-commentare e commentare l'uno o l'altro.
		   I risultati ottenuti da entrambi sono descritti nella relazione in .pdf. 
		   Sono state inoltre commentate le funzioni di stampa, essendo esse approssimative, seppur può essere utile visualizzare entrambi gli alberi.

- Un Notebook Jupyter, nel quale ho testato la funzione di prune su un piccolo df, forzando alcuni parametri. NB: cambiare il percorso del dtree_helper.py all'interno del file Jupyter

- La cartella contenente i datasets reperiti sul sito: https://archive.ics.uci.edu/ml/datasets.php


NB: Il codice qui riprodotto è stato quanto più prodotto autonomamente, dando comunque uno sguardo a lavori altrui in rete. Ho trovato molto esplicativo ed interessante tale fonte: 

    https://automaticaddison.com/iterative-dichotomiser-3-id3-algorithm-from-scratch/
    
