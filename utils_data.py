label2id = {"chemprot_blue": {"false": 0, "CPR:3": 1, "CPR:4": 2, "CPR:5": 3, "CPR:6": 4, "CPR:9": 5},
            "chemprot_blurb": {"false": 0, "CPR:3": 1, "CPR:4": 2, "CPR:5": 3, "CPR:6": 4, "CPR:9": 5},
            "ddi_blue": {"DDI-false": 0, "DDI-mechanism": 1, "DDI-effect": 2, "DDI-advise": 3, "DDI-int": 4},
            "ddi_blurb": {"DDI-false": 0, "DDI-mechanism": 1, "DDI-effect": 2, "DDI-advise": 3, "DDI-int": 4},
            "i2b2": {"false": 0, "TeCP": 1, "TrCP": 2, "TrAP": 3, "TeRP": 4, "PIP": 5, "TrWP": 6, "TrIP": 7, "TrNAP": 8},
            "i2b2_modified": {"false": 0, "TeCP": 1, "TrCP": 2, "TrAP": 3, "TeRP": 4, "PIP": 5, "TrWP": 6, "TrIP": 7, "TrNAP": 8},
            }

class_weights = {"drugprot":[1.358,45.340,98.397,2232.586,4980.385,66.610,28.814,48.717,46.985,12.023,73.158,70.375,32.324,2697.708],
                 "chemprot":[1.461,17.083,5.831,75.838,55.830,18.427],
                 "chemprot_blurb":[1.295,23.856,8.098,104.249,78.755,24.807],
                 "chemprot_blue":[1.271,25.339,8.645,112.486,82.809,26.768],
                 "bbrel":[1.313,4.195],
                 "ddi_blue":[1.185,19.851,15.494,29.667,128.623],
                 "ddi_blurb":[1.163,21.751,17.023,34.891,142.815],
                 "i2b2":[1.169,140.886,120.978,25.210,22.417,26.035,927.500,436.471,359.032],
                 "i2b2_modified":[1.163,134.450,117.152,25.041,22.185,30.399,834.708,408.837,385.250],}

bert_names_to_versions = {"pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                          "biobert": "dmis-lab/biobert-base-cased-v1.2",
                          "scibert": "allenai/scibert_scivocab_uncased",
                          "bluebert": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                          "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
                          "biolinkbert": "michiyasunaga/BioLinkBERT-base",}

number_of_labels = {"bbrel":2,
                    "chemprot_blue":6,
                    "chemprot_blurb":6,
                    "drugprot":14,
                    "ddi_blue":5,
                    "ddi_blurb":5,
                    "i2b2":9,
                    "i2b2_modified":9,}
