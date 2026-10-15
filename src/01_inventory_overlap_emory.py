import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
"""Phase 0, pass 2 (read-only except results/harvi/).
(a) HARVI patients present in FairECG's S5-eng pretraining corpus (patient_id 'mimic:<subject_id>')
(b) Emory (HEEDB I0006) HF phenotype counts under the draft pre-specified definition
(c) Emory manifest <-> HEEDB metadata link, and EchoNext LVEF availability (secondary)
"""
import numpy as np, pandas as pd, json, os
HE = f"{_PR}"; OUT = f"{HE}/results/harvi"; E = f"{HE}/data/heedb/I0006"
R = {}
hp = np.load(f"{OUT}/harvi_pairs_folds.npz")
hs = set(np.unique(hp["subject_id"]).astype(str))
meta = pd.read_parquet(f"{HE}/_jobs/scaling_s5_eng/shard_meta.parquet", columns=["patient_id", "cohort"])
pm = set(meta.loc[meta.cohort == "mimic", "patient_id"].astype(str).str.replace("mimic:", "", regex=False))
R["s5eng_mimic_patients"] = len(pm)
R["harvi_patients"] = len(hs); R["harvi_patients_in_pretraining"] = len(hs & pm)
inpre = np.isin(hp["subject_id"].astype(str), list(pm))
R["harvi_pairs_in_pretraining"] = int(inpre.sum()); np.save(f"{OUT}/harvi_pair_in_pretraining.npy", inpre)

# ---- (b) Emory phenotypes --------------------------------------------------------
md = pd.read_csv(f"{E}/metadata/metadata.csv", dtype={"BDSPPatientID": str}, low_memory=False,
                 usecols=["BDSPPatientID", "FileName", "ECGAcquisitionTime", "LastKnownVisitDate", "AgeAtAcquisition", "Sex", "PatientRace"])
md["BDSPPatientID"] = md.BDSPPatientID.str.replace(r"\.0$", "", regex=True)
md["AgeAtAcquisition"] = pd.to_numeric(md.AgeAtAcquisition, errors="coerce") / 365.25   # stored in days
md["t"] = pd.to_datetime(md.ECGAcquisitionTime, errors="coerce"); md["last"] = pd.to_datetime(md.LastKnownVisitDate, errors="coerce")
R["emory_ecgs"] = int(len(md)); R["emory_patients"] = int(md.BDSPPatientID.nunique())
codes = []
for ch in pd.read_csv(f"{E}/ICD_codes/icd10_codes.csv", dtype={"BDSPPatientID": str, "DIAGNOSIS_ICD10_CD": str},
                      usecols=["BDSPPatientID", "RECORDED_DT", "DIAGNOSIS_ICD10_CD"], chunksize=5_000_000):
    c = ch.DIAGNOSIS_ICD10_CD.str.replace(".", "", regex=False).str.upper()
    k = c.str.startswith("I50")
    if k.any():
        x = ch[k].copy(); x["c"] = c[k]; codes.append(x)
icd = pd.concat(codes); icd["BDSPPatientID"] = icd.BDSPPatientID.str.replace(r"\.0$", "", regex=True)
R["id_overlap_meta_icd"] = int(len(set(md.BDSPPatientID) & set(icd.BDSPPatientID))); icd["dt"] = pd.to_datetime(icd.RECORDED_DT, errors="coerce")
icd["ph"] = np.select([icd.c.str.startswith("I502"), icd.c.str.startswith("I503"), icd.c.str.startswith("I504")], ["hfref", "hfpef", "mixed"], "unspec")
R["emory_hf_code_rows"] = {k: int(v) for k, v in icd.ph.value_counts().items()}
per = icd.groupby("BDSPPatientID").ph.agg(lambda s: set(s))
def assign(s):
    sp = s - {"unspec"}
    if sp == {"hfref"}: return "hfref"
    if sp == {"hfpef"}: return "hfpef"
    if sp: return "mixed"
    return "unspec"
pheno = per.map(assign)
R["emory_age_ok"] = float(md.AgeAtAcquisition.median())
R["emory_hf_patients_by_phenotype"] = {k: int(v) for k, v in pheno.value_counts().items()}
first = icd[icd.ph.isin(["hfref", "hfpef"])].groupby("BDSPPatientID").dt.min()
anyhf = set(per.index)
# draft definition: case ECG = the ECG closest to the first subtype-specific code within [-180, +30] days
mm = md.merge(first.rename("first_dx"), left_on="BDSPPatientID", right_index=True, how="left")
mm["dd"] = (mm.t - mm.first_dx).dt.days
R["emory_case_window_counts"] = {w: {k:int(v) for k,v in mm[mm.first_dx.notna() & mm.dd.between(-w, 30)].drop_duplicates("BDSPPatientID").BDSPPatientID.map(pheno).value_counts().items()} for w in (30, 90, 180, 365)}
case = mm[mm.first_dx.notna() & mm.dd.between(-180, 30)].copy(); case["ad"] = case.dd.abs()
case = case.sort_values("ad").groupby("BDSPPatientID").head(1)
case["phenotype"] = case.BDSPPatientID.map(pheno)
R["emory_case_ecgs_by_phenotype"] = {k: int(v) for k, v in case.phenotype.value_counts().items()}
# controls: no I50 code ever, >= 365 days follow-up after the ECG; one random ECG per patient (seed 0)
ctl = md[~md.BDSPPatientID.isin(anyhf) & ((md["last"] - md.t).dt.days >= 365)]
ctl = ctl.sample(frac=1.0, random_state=0).groupby("BDSPPatientID").head(1)
R["emory_control_ecgs"] = int(len(ctl))
R["emory_age_median_case_ctl"] = [float(case.AgeAtAcquisition.median()), float(ctl.AgeAtAcquisition.median())]
case[["BDSPPatientID", "FileName", "phenotype", "dd", "AgeAtAcquisition", "Sex", "PatientRace"]].to_parquet(f"{OUT}/emory_cases_draft.parquet")
ctl[["BDSPPatientID", "FileName", "AgeAtAcquisition", "Sex", "PatientRace"]].to_parquet(f"{OUT}/emory_controls_draft.parquet")

# ---- (c) links ---------------------------------------------------------------------
em = pd.read_parquet(f"{HE}/manifests/manifest_emory.parquet")
R["emory_manifest_cols"] = list(em.columns); R["emory_manifest_head"] = em.head(2).astype(str).to_dict("records")
en = pd.read_csv(f"{HE}/data/echonext/ECHONEXT/echonext_metadata_100k.csv", usecols=["patient_key", "lvef_value", "split"])
R["echonext_lvef"] = {"nonnull": int(en.lvef_value.notna().sum()), "le40": int((en.lvef_value <= 40).sum()),
                      "ge50": int((en.lvef_value >= 50).sum()), "patients": int(en.patient_key.nunique())}
json.dump(R, open(f"{OUT}/inventory2.json", "w"), indent=1, default=str)
print("INVENTORY2 DONE")
