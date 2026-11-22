"""Tissue classes and how each public source's labels map onto them."""

CLASSES = ["tumour", "normal_epithelium", "stroma", "immune", "necrosis", "fat", "vessels",
           "muscle", "other"]
NONTUMOUR = CLASSES[1:]

# Kather NCT-CRC-HE-100K / CRC-VAL-HE-7K codes
KATHER_MAP = {"TUM": "tumour", "NORM": "normal_epithelium", "STR": "stroma", "LYM": "immune",
              "DEB": "necrosis", "ADI": "fat", "MUS": "muscle", "BACK": "other", "MUC": "other"}

# BCSS mask codes (Amgad 2019, gtruth_codes.tsv); None = tile not used. Checked on download.
BCSS_MAP = {
    0: None,                  # outside_roi
    1: "tumour",              # tumor
    2: "stroma",              # stroma
    3: "immune",              # lymphocytic_infiltrate
    4: "necrosis",            # necrosis_or_debris
    5: "other",               # glandular_secretions
    6: "necrosis",            # blood (our class is necrosis or haemorrhage)
    7: None,                  # exclude
    8: None,                  # metaplasia_NOS
    9: "fat",                 # fat
    10: "immune",             # plasma_cells
    11: "immune",             # other_immune_infiltrate
    12: "other",              # mucoid_material
    13: "normal_epithelium",  # normal_acinus_or_duct
    14: None,                 # lymphatics
    15: None,                 # undetermined
    16: "other",              # nerve
    17: "other",              # skin_adnexa
    18: "vessels",            # blood_vessel
    19: None,                 # angioinvasion
    20: "tumour",             # dcis (as in the paper's breast classes)
    21: "other",              # other
}

# TCGA-UT cancer-type labels (Zenodo archive names) -> our cohort type codes.
_TCGAUT = {
    "bladder_urothelial_carcinoma": "BLCA",
    "breast_invasive_carcinoma": "BRCA",
    "cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma": "CESC",
    "colon_adenocarcinoma": "COAD",
    "rectum_adenocarcinoma": "COAD",  # colorectal tissue; READ is not in our cohort
    "esophageal_carcinoma": "ESCA",
    "head_and_neck_squamous_cell_carcinoma": "HNSC",
    "kidney_renal_clear_cell_carcinoma": "KIRC",
    "kidney_renal_papillary_cell_carcinoma": "KIRP",
    "liver_hepatocellular_carcinoma": "LIHC",
    "lung_adenocarcinoma": "LUAD",
    "lung_squamous_cell_carcinoma": "LUSC",
    "ovarian_serous_cystadenocarcinoma": "OV",
    "pancreatic_adenocarcinoma": "PAAD",
    "prostate_adenocarcinoma": "PRAD",
    "stomach_adenocarcinoma": "STAD",
    "thyroid_carcinoma": "THCA",
    "uterine_corpus_endometrial_carcinoma": "UCEC",
}


def _norm(label):
    return "_".join(str(label).strip().lower().replace("-", " ").replace("_", " ").split())


def tcgaut_type(label):
    """Our cohort type code for a TCGA-UT label, or None for types outside the 17."""
    return _TCGAUT.get(_norm(label))


# GTEx tissues standing in for each cancer type's organ (GTEx SMTSD names; checked against
# the IDC index when fetching). HNSC has no oral mucosa in GTEx: squamous oesophageal mucosa
# and minor salivary gland stand in.
GTEX_TISSUES = {
    "BLCA": ["Bladder"], "BRCA": ["Breast - Mammary Tissue"],
    "CESC": ["Cervix - Ectocervix", "Cervix - Endocervix"], "COAD": ["Colon - Transverse"],
    "ESCA": ["Esophagus - Mucosa"], "HNSC": ["Esophagus - Mucosa", "Minor Salivary Gland"],
    "KIRC": ["Kidney - Cortex"], "KIRP": ["Kidney - Cortex"], "LIHC": ["Liver"],
    "LUAD": ["Lung"], "LUSC": ["Lung"], "OV": ["Ovary"], "PAAD": ["Pancreas"],
    "PRAD": ["Prostate"], "STAD": ["Stomach"], "THCA": ["Thyroid"], "UCEC": ["Uterus"],
}

# Fixed before results: where a class that fails its bar goes (spec, Pass bars).
MERGE = {"vessels": "stroma", "muscle": "stroma", "immune": "stroma", "necrosis": "other",
         "fat": "other", "normal_epithelium": "other", "stroma": "other"}
