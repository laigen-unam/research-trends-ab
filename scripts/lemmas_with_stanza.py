from stanza import download
from stanza import Pipeline
from time import time
import string
import os

def stanza(input):

    t0 = time()

    try:
        nlp = Pipeline(lang='en', processors='tokenize,lemma',tokenize_no_ssplit=True)
    except:
        download(lang="en",package=None,processors={"tokenize":"combined", "lemma":"combined"})
        nlp = Pipeline(lang='en', processors='tokenize,lemma',tokenize_no_ssplit=True)


    abstracts= []
    # CFMC: We add title lemmatization
    titles = []
    with open(input,"r") as handle:
        for line in handle:
            pmid = ""
            line = line.strip("\r\n")     #Se eliminan los saltos de linea y retornos de carro
            line = line.split("\t")    #Se divide cada linea por tabulador
            pmid = line[0]
            # CFMC: We add title lemmatization
            line_tit = line[1]  #Se selecciona la posicion 1 de cada linea, que corresponde a la posicion de los titulos
            line_tit = pmid + " " + line_tit
            line = line[2]    #Se selecciona la posicion 2 de cada linea, que corresponde a la posicion de los abstracts
            line = pmid + " " + line
            line = line.replace("\t","")
            ln = line.split(" ")     #Se separan las palabras
            table = str.maketrans('', '', string.punctuation)     #Para eliminar los signos de puntuacion
            abstract = [l.translate(table) for l in ln]     #Se eliminan los signos de puntuacion
            abstract = " ".join(abstract)     #Se añade un espacio donde se encontraban los signos de puntuacion
            abstract = abstract+"\n\n"
            abstracts.append(abstract)
            # CFMC: We add title lemmatization
            line_tit = line_tit.replace("\t","")
            ln = line_tit.split(" ")     #Se separan las palabras
            table = str.maketrans('', '', string.punctuation)     #Para eliminar los signos de puntuacion
            title = [l.translate(table) for l in ln]     #Se eliminan los signos de puntuacion
            title = " ".join(title)     #Se añade un espacio donde se encontraban los signos de puntuacion
            title = title+"\n\n"
            titles.append(title)

        print("Length abstracts: {}".format(len(abstracts)))
        print("Length titles: {}".format(len(titles)))
        abs_lemma = nlp(abstracts)
        print("Length abs_lemma: {}".format(len(abs_lemma.sentences)))
        # CFMC: We add title lemmatization
        tit_lemma = nlp(titles)
        print("Length tit_lemma: {}".format(len(tit_lemma.sentences)))

        print("Creating output file...")
        final_file = []      #Se crea una lista donde se guardara el archivo final a escribir
        with open(input, mode='r') as iFile:      #Se abre nuevamente el archivo de abstracts para crear el nuevo
            # CFMC: We add title lemmatization
            i = 0
            for line in iFile:     #Se lee cada linea del archivo de abstracts
                pmid = ""
                line = line.strip("\r\n")     #Se eliminan los saltos de linea y retornos de carro
                line = line.split("\t")    #Se divide cada linea por tabulador
                pmid = line[0]
                pmid_abs_lemmatized = abs_lemma.sentences[i].words[0].lemma
                pmid_tit_lemmatized = tit_lemma.sentences[i].words[0].lemma
                if pmid_tit_lemmatized != pmid_abs_lemmatized:
                    print("ERROR: PMID Mismatch between lemmatized title and lemmatized abstracts")
                    print("Run aborted!")
                    quit()
                if pmid != pmid_tit_lemmatized:
                    print("PMID Mismatch between abstracts and lemmatized title")
                    i += 1
                    continue
                if pmid != pmid_abs_lemmatized:
                    print("PMID Mismatch between abstracts and lemmatized abstracts")
                    i += 1
                    continue
                abs_sentences_string = " ".join([word.lemma for word in abs_lemma.sentences[i].words[1:] if word.lemma is not None]) # CFMC: para evitar error en join con valores None
                line[2] = abs_sentences_string      #Se coloca el abstract lematizado en la posicion 2 de la linea
                # CFMC: We add title lemmatization
                tit_sentences_string = " ".join([word.lemma for word in tit_lemma.sentences[i].words[1:] if word.lemma is not None]) # CFMC: para evitar error en join con valores None
                line[1] = tit_sentences_string      #Se coloca el abstract lematizado en la posicion 2 de la linea
                final_file.append(line)      #Se fguarda la linea en el archivo final
                i += 1
        print("Done!\n")


        print("Saving output file...")
        with open(os.path.splitext(input)[0]+"_lemmatized.txt", mode="w") as oFile:  #Se abre el archivo de salida
            for i in final_file:
                oFile.write("\t".join([j for j in i]))
                oFile.write("\n")

    return(time()-t0)
# Publications for clustering
input_path = "abs.txt"
stanza(input_path)