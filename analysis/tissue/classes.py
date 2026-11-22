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
    "colon_rectum_adenocarcinoma": "COAD",  # Hugging Face TCGA-UT merges COAD and READ
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


# GTEx tissues standing in for each cancer type's organ, by their IDC name
# (sm_index primaryAnatomicStructure_CodeMeaning, collection gtex, checked 2026-11-22).
# HNSC has no oral mucosa in GTEx: squamous oesophageal mucosa and minor salivary gland stand in.
GTEX_TISSUES = {
    "BLCA": ["Urinary bladder"], "BRCA": ["Breast"], "CESC": ["Exocervix", "Endocervix"],
    "COAD": ["Transverse colon"], "ESCA": ["Esophageal mucous membrane"],
    "HNSC": ["Esophageal mucous membrane", "Minor salivary gland"],
    "KIRC": ["Renal cortex"], "KIRP": ["Renal cortex"], "LIHC": ["Liver"], "LUAD": ["Lung"],
    "LUSC": ["Lung"], "OV": ["Ovary"], "PAAD": ["Pancreas"], "PRAD": ["Prostate"],
    "STAD": ["Stomach"], "THCA": ["Thyroid"], "UCEC": ["Uterus"],
}

# Fixed before results: where a class that fails its bar goes (spec, Pass bars).
MERGE = {"vessels": "stroma", "muscle": "stroma", "immune": "stroma", "necrosis": "other",
         "fat": "other", "normal_epithelium": "other", "stroma": "other"}

# TIGER ROI tissue masks (wsirois, tissue-cells): 1 invasive tumour, 2 tumour-associated stroma,
# 3 in-situ tumour, 4 healthy glands, 5 necrosis (not in situ), 6 inflamed stroma, 7 rest.
# Inflamed stroma could be stroma or immune, and rest is mixed, so both are dropped.
TIGER_MAP = {0: None, 1: "tumour", 2: "stroma", 3: "tumour", 4: "normal_epithelium",
             5: "necrosis", 6: None, 7: None}

# Brummer et al. 2023 ccRCC tiles (Zenodo 7898308, CC BY 4.0), by folder. 'blood' counts as haemorrhage
# (necrosis class); 'empty' is background and 'other' mixes fat with other tissue, so both are dropped.
BRUMMER_MAP = {"cancer": "tumour", "normal": "normal_epithelium", "stroma": "stroma",
               "blood": "necrosis", "empty": None, "other": None}


def brummer_group(filename):
    """(split group, patient) of a Brummer tile: the TCGA site and barcode, or the Helsinki slide."""
    if filename.startswith("TCGA-"):
        barcode = filename[:12]
        return barcode.split("-")[1], barcode
    slide = filename.split(".mrxs")[0]
    return f"brummer:{slide}", slide

# SegPath (Komura et al. 2023, Zenodo, CC BY-NC-SA 4.0; immunofluorescence masks on H&E at 0.22 um/px).
# Only CD45RB leukocytes are used: a 64 um tile is immune when the marker covers >= 20% of it (peek grids:
# dense lymphocytes and plasma cells). aSMA is not used (it also marks myofibroblasts and myoepithelium;
# tiles >= 80% covered still include stroma and tumour-like cells); ERG is not used (endothelium covers at
# most ~3% of a tile, so no tile is a 'vessel').
SEGPATH_IMMUNE_MIN = 0.20


def segpath_slide(filename):
    """SegPath slide id: the number after the marker name ('CD45RB_Leukocyte_205_..._HE.png' -> '205')."""
    return filename.split("/")[-1].split("_")[2]
