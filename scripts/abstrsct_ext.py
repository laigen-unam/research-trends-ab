# -*- coding: utf-8 -*-

from Bio import Entrez

def download_abstracts(input_file, mail, output_file, report_file):
    ##read archive for pmids
    with open(input_file, "r") as arc:
        pmids = [line.split('\n')[0] for line in arc]

    # email for contact in case of excesive demand prior blocking
    Entrez.email = mail

    handle = Entrez.efetch(db="pubmed", id=','.join(map(str, pmids)),
                           rettype="xml", retmode="text")
    records = Entrez.read(handle)

    # save pmids without abstract for future report
    withoutabs = []

    for pmid in records['PubmedArticle']:
        if 'Abstract' not in pmid['MedlineCitation']['Article']:
            withoutabs.append(str(pmid['MedlineCitation']['PMID']))

    # create string for all the authors as a dict key
    for pmid in records['PubmedArticle']:
        if 'Abstract' and 'AuthorList' in pmid['MedlineCitation']['Article']:
            s = ''
            for autor in pmid['MedlineCitation']['Article']['AuthorList']:
                if 'ForeName' in autor.keys():
                    s += f"{autor['ForeName']} {autor['LastName']}, "
                elif 'Initials' in autor.keys():
                    s += f"{autor['LastName']} {autor['Initials']}, "
                elif 'CollectiveName' in autor.keys():
                    s += f"{autor['CollectiveName']}, "
                else:
                    s += f"{autor['LastName']}, "
            s = s[:-2]
        else:
            s = 'No authors listed'
        pmid['Autors'] = s

    f = open(output_file, 'w')

    for pmid in records['PubmedArticle']:
        keys = pmid['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate'].keys()
        if 'Month' in keys and 'Abstract' in pmid['MedlineCitation']['Article']:
            f.write(pmid['MedlineCitation']['PMID'] + '\t' +
                    pmid['MedlineCitation']['Article']['ArticleTitle'] + '\t' +
                    str(pmid['MedlineCitation']['Article']['Abstract']['AbstractText'][0]).replace("\n", " ") + '\t' +
                    pmid['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'] + ' ' +
                    pmid['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Month'] + '\t' +
                    pmid['Autors'] + '\t' +
                    pmid['MedlineCitation']['Article']['Journal']['ISOAbbreviation'] +
                    '\n')
            continue
        elif 'MedlineDate' in keys and 'Abstract' in pmid['MedlineCitation']['Article']:
            f.write(pmid['MedlineCitation']['PMID'] + '\t' +
                    pmid['MedlineCitation']['Article']['ArticleTitle'] + '\t' +
                    str(pmid['MedlineCitation']['Article']['Abstract']['AbstractText'][0]).replace("\n", " ") + '\t' +
                    pmid['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['MedlineDate'] + '\t' +
                    pmid['Autors'] + '\t' +
                    pmid['MedlineCitation']['Article']['Journal']['ISOAbbreviation'] +
                    '\n')
            continue
        elif 'Month' not in keys and 'Abstract' in pmid['MedlineCitation']['Article']:
            f.write(pmid['MedlineCitation']['PMID'] + '\t' +
                    pmid['MedlineCitation']['Article']['ArticleTitle'] + '\t' +
                    str(pmid['MedlineCitation']['Article']['Abstract']['AbstractText'][0]).replace("\n", " ") + '\t' +
                    pmid['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'] + '\t' +
                    pmid['Autors'] + '\t' +
                    pmid['MedlineCitation']['Article']['Journal']['ISOAbbreviation'] +
                    '\n')

    f.close()

    f = open(output_file, 'r')

    pmids_retrieved = [line.split('\t')[0] for line in f]

    not_found = []

    for id in pmids:
        if id not in pmids_retrieved:
            if id not in withoutabs:
                not_found.append(id)

    f.close()

    g = open(report_file, 'w')

    g.write('total_pmids recieved\tpmids_retrieved\tpmids_without_abstracts\tpmids_not_found\n' +
            str(len(pmids)) + '\t' + str(len(pmids_retrieved)) + '\t' + str(len(withoutabs)) + '\t' + str(
        len(not_found)))

    g.close()

    return [str(len(pmids)), str(len(pmids_retrieved)), str(len(withoutabs)), str(len(not_found))]


i_file = "pmid-Acinetobac-set-2022-2023.txt"
i_mail = "cmendezc@ccg.unam.mx"
o_file = "abs-2022-2023.txt"
r_file = "abs_downloaded-2022-2023.txt"
download_result = download_abstracts(i_file, i_mail, o_file, r_file)
print("PMIDs input", download_result[0])
print("PMIDs retrieved", download_result[1])
print("PMIDS without abstract", download_result[2])
print("PMIDs not found", download_result[3])
