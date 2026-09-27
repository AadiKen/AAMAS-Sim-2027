"""Materialize source/mapping audit from published evidence and saved CFD only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "stage3_results/spec_a/kcs/coarse_first"
OUT = BASE / "benchmark_mapping_audit"
SOURCES = ROOT / "stage3_results/kcs-validation/sources"


def field(status, value=None, evidence=None):
    return {"status": status, "value": value, "evidence": evidence}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = SOURCES / "sung_park_2015.pdf"
    cmt_figure = OUT / "kim_2009_fig7_cmt_hp.png"
    harmonic_source = BASE / "harmonic_yaw_postprocess_audit/corrected_harmonics.json"
    corrected = json.loads(harmonic_source.read_text())
    matrix = {
      "status_legend": ["KNOWN", "DERIVED", "CONFLICTING", "MISSING"],
      "sources": [
       {"id": "sung_park_2015", "title": "Prediction of Ship Manoeuvring Performance Based on Virtual Captive Model Tests (Sung & Park, 2015)",
        "provenance": {"url": "https://oak.go.kr/repository/journal/17389/DHJSCN_2015_v52n5_407.pdf",
                       "local_file": str(sp), "sha256": sha(sp),
                       "extracted_text_file": str(SOURCES / "sung_park_2015.txt"),
                       "extracted_text_sha256": sha(SOURCES / "sung_park_2015.txt"),
                       "access": "complete_repository_PDF"},
        "fields": {
         "vessel_configuration": field("KNOWN", "bare hull for Table 6", "p408 Table 1; p412 Table 6"),
         "model_scale": field("KNOWN", 40, "p409 Table 3"),
         "L_m": field("KNOWN", 5.750, "p409 Table 3"),
         "B_m": field("KNOWN", .805, "p409 Table 3"),
         "T_m": field("CONFLICTING", {"printed": .207, "fullscale_over_40": .270}, "p409 Table 3"),
         "displacement_m3": field("KNOWN", .8130, "p409 Table 3"),
         "mass_kg": field("MISSING"),
         "CG_LCG_m": field("MISSING", evidence="p408 Eq1 uses CG, but coordinates not given"),
         "moment_origin": field("MISSING", evidence="p408 equations at CG; load-cell origin not specified"),
         "axes_signs": field("MISSING", evidence="signed experimental phase/axis map not explicit"),
         "normalization": field("KNOWN", "L,T,initial U; N'=N/(0.5 rho L^2 T U^2)", "p408 before Eqs2-4"),
         "speed_mps": field("DERIVED", 1.953, "Fn=.260 and Lpp=5.750, p409 Table 3; case setting"),
         "Fn": field("KNOWN", .260, "p409 Table 3"),
         "Re": field("KNOWN", 9.851e6, "p409 Table 3"),
         "drift_yaw_conditions": field("KNOWN", "pure yaw beta=0, r'max=.1-.6", "p408 Table 1"),
         "motion_equation": field("MISSING", evidence="pure-yaw trajectory described but exact time equation absent"),
         "frequency_hz": field("MISSING", evidence="0.08 Hz in CFD case is inferred from Fig6, not tabulated"),
         "harmonic_phase": field("MISSING", evidence="out-phase graph in p412 Fig6 lacks signed cosine zero definition"),
         "inertial_corrections": field("KNOWN", "inertial and added-mass terms separated in equations; exact raw gauge correction not given", "p408 Eqs1,3,4"),
         "load_type": field("KNOWN", "fitted coefficients, not raw time history", "p412 Table 6"),
         "numerical_data": field("KNOWN", {"N_r_prime": -.0462, "N_rrr_prime": -.0313}, "p412 Table 6"),
         "plots": field("KNOWN", "pure-yaw first out-phase and time traces", "p412 Fig6"),
         "time_histories": field("MISSING")}},
       {"id": "shin_choi_2011", "title": "Prediction of Maneuverability of KCS Using Captive Model Test",
        "provenance": {"doi": "10.3744/SNAK.2011.48.5.465",
                       "url": "https://koreascience.or.kr/article/JAKO201132833920245.page",
                       "pdf_url": "https://koreascience.or.kr/article/JAKO201132833920245.pdf",
                       "local_file": None, "sha256": None,
                       "access": "publisher landing and abstract verified; PDF endpoint did not yield local bytes"},
        "fields": {
         "vessel_configuration": field("MISSING", evidence="abstract does not establish bare hull vs appended per CMT load"),
         "model_scale": field("DERIVED", "approximately 2 m class, not 1:40", "publisher abstract"),
         "L_m": field("MISSING"), "B_m": field("MISSING"), "T_m": field("MISSING"),
         "displacement_m3": field("MISSING"), "mass_kg": field("MISSING"),
         "CG_LCG_m": field("MISSING"), "moment_origin": field("MISSING"),
         "axes_signs": field("MISSING"), "normalization": field("MISSING"),
         "speed_mps": field("MISSING"), "Fn": field("MISSING"), "Re": field("MISSING"),
         "drift_yaw_conditions": field("KNOWN", "CMT and PMM performed; individual states unavailable in abstract", "publisher abstract"),
         "motion_equation": field("MISSING"), "frequency_hz": field("MISSING"),
         "harmonic_phase": field("MISSING"), "inertial_corrections": field("MISSING"),
         "load_type": field("MISSING"), "numerical_data": field("MISSING"),
         "plots": field("MISSING"), "time_histories": field("MISSING")}},
       {"id": "kim_2009_cpmc", "title": "Prediction of Maneuverability of KCS by CPMC Captive Model Test",
        "provenance": {"doi": "10.3744/SNAK.2009.46.6.553",
                       "url": "https://www.researchgate.net/publication/250272414_Prediction_of_Maneuverability_of_KCS_by_CPMC_Captive_Model_Test",
                       "local_file": None, "sha256": None,
                       "archived_figure": str(cmt_figure), "archived_figure_sha256": sha(cmt_figure),
                       "access": "author-uploaded full text readable in indexed HTML; Figure 7 archived; PDF bytes not locally archived"},
        "fields": {
         "vessel_configuration": field("KNOWN", "CMT H+P; rudder contribution removed, propeller remains", "p557 Fig7 and text"),
         "model_scale": field("KNOWN", {"MOERI":65.833,"NMRI":75.5}, "p555 Table 1"),
         "L_m": field("KNOWN", {"MOERI":3.4937,"NMRI":3.0464}, "p555 Table 1"),
         "B_m": field("KNOWN", {"MOERI":.4891,"NMRI":.4265}, "p555 Table 1"),
         "T_m": field("KNOWN", {"MOERI":.1641,"NMRI":.1430}, "p555 Table 1"),
         "displacement_m3": field("KNOWN", {"MOERI":.1824,"NMRI":.1209}, "p555 Table 1"),
         "mass_kg": field("MISSING"), "CG_LCG_m": field("MISSING", evidence="x_G occurs in p555 Eq1; numeric value unavailable"),
         "moment_origin": field("MISSING", evidence="p555 Eq1 has x_G; Fig7 gauge origin unverified"),
         "axes_signs": field("KNOWN", "body axes drawn in p555 Fig3; exact mapping to BCOD not established", "p555 Fig3"),
         "normalization": field("KNOWN", "Force'=Force/(0.5 rho U^2 L^2); Moment'=Moment/(0.5 rho U^2 L^3)", "p556 Eq5"),
         "speed_mps": field("KNOWN", {"MOERI":1.522,"NMRI":1.1}, "p555 Table1"),
         "Fn": field("DERIVED", {"MOERI":.260,"NMRI":.201}, "p555 Table1 U,L"),
         "Re": field("MISSING"),
         "drift_yaw_conditions": field("KNOWN", "MOERI CMT r'=±.15,.2,.3,.4,.5,.6,.7; propeller+rudder turning family distinct", "p556 Table2"),
         "motion_equation": field("KNOWN", "steady circular-motion test", "p556 Table2"),
         "frequency_hz": field("MISSING", evidence="not applicable to steady CMT"),
         "harmonic_phase": field("MISSING", evidence="not applicable to steady CMT"),
         "inertial_corrections": field("MISSING", evidence="measured-load correction not established"),
         "load_type": field("KNOWN", "Fig7 measured/fitted H+P values after rudder subtraction", "p557 Fig7 and text"),
         "numerical_data": field("MISSING", evidence="no numeric force/moment table at r'=.2,.4; figure only"),
         "plots": field("KNOWN", "CMT Y',N' H+P curves", "p557 Fig7"),
         "time_histories": field("MISSING")}},
       {"id": "simman_2014", "title": "SIMMAN 2014 captive datasets",
        "provenance": {"url": "https://simman2014.dk/about-simman2014/model-test-plan/",
                       "news_url": "https://simman2014.dk/about-simman2014/latest-news/",
                       "local_file": None, "sha256": None,
                       "access": "official plan and news available; underlying FTP dataset not acquired"},
        "fields": {
          **{k: field("MISSING") for k in ("B_m","T_m",
                   "displacement_m3","mass_kg","CG_LCG_m","moment_origin","axes_signs","normalization",
                   "speed_mps","Re","drift_yaw_conditions","motion_equation","frequency_hz",
                   "harmonic_phase","inertial_corrections","load_type","numerical_data","plots","time_histories")},
          "vessel_configuration": field("KNOWN", "FORCE appended and bare deep-water PMM; NMRI 3DOF appended deep-water CMT", "official model-test plan, KCS row"),
          "model_scale": field("KNOWN", {"FORCE_PMM":52.667,"NMRI_CMT":75.5}, "official model-test plan, KCS row"),
          "L_m": field("KNOWN", {"FORCE_PMM":4.3671,"NMRI_CMT":3.0464}, "official model-test plan, KCS row"),
          "Fn": field("KNOWN", {"FORCE_PMM":.26,"NMRI_CMT":.20}, "official model-test plan, KCS row")}}
      ]}
    write("benchmark_source_matrix.json", matrix)
    cmt = {"comparison_family":"steady_cmt", "admitted_rows": [],
           "candidate_source":"kim_2009_cpmc", "candidate_states_rprime":[.2,.4],
           "rejection_reasons":["2009 Fig7 is H+P after rudder subtraction; CFD is bare hull",
                                "2009 model scale/speed differ and reference origin is missing",
                                "2009 digitized points are H+P, not bare-hull loads",
                                "Shin-Choi 2011 full text/numeric CMT data not acquired"],
           "digitized_rows": [
             {"source":"kim_2009_p557_Fig7_MOERI", "r_prime":.2,"N_prime":-.00035,
              "absolute_uncertainty_N_prime":.00008,"admission":"REJECTED_HULL_PLUS_PROPELLER"},
             {"source":"kim_2009_p557_Fig7_MOERI", "r_prime":.4,"N_prime":-.00089,
              "absolute_uncertainty_N_prime":.00008,"admission":"REJECTED_HULL_PLUS_PROPELLER"}],
           "digitization_calibration": {"figure_file":str(cmt_figure), "sha256":sha(cmt_figure),
                 "pixel_width":586,"pixel_height":488,
                 "x_axis_points_pixel_rprime":[[76,-.8],[296,0],[516,.8]],
                 "y_axis_points_pixel_Nprime":[[232,0],[90,.002],[374,-.002]],
                 "MOERI_marker_centers_estimated_pixel":[[351,257],[406,295]],
                 "source":"Kim et al. 2009 p557 Fig7, author-uploaded ResearchGate figure"},
           "digitization_uncertainty": "±0.00008 N' from roughly ±5-pixel marker/line-center ambiguity at ~71 px per .001 N'"}
    write("cmt_reference.json", cmt)
    harmonic_ref = {"comparison_family":"harmonic_pmm", "admitted_harmonic_rows": [],
       "published_coefficients": {"source":"sung_park_2015_p412_Table6", "N_r_prime":-.0462,
                                  "N_rrr_prime":-.0313, "status":"KNOWN"},
       "absolute_load_harmonics": None,
       "blocking_fields":["experimental_CG_coordinates", "load_cell_moment_origin", "signed_phase_convention",
                          "exact_pure_yaw_frequency", "draft_0.207_vs_0.270", "raw_vs_inertial_corrected_load"]}
    write("harmonic_reference.json", harmonic_ref)
    steady = []
    for name, rprime in (("diagnostic_yaw_0_+0.1",.1),("05_yaw_0_+0.2",.2),("06_yaw_0_+0.4",.4)):
        path = BASE/name/"run_result.json"
        data = json.loads(path.read_text())
        fy, mz = data["qualification"]["mean_foam"][1], data["qualification"]["mean_foam"][5]
        steady.append({"case":name,"r_prime":rprime,"Y_n_physical_frd":-fy,
                       "N_nm_physical_frd_at_case_origin":-mz,
                       "foam_reference_origin_m":[0,0,0],"case_cg_frd_m":[0,0,0],
                       "qualification_status":data["qualification"]["status"],
                       "source_file":str(path),"source_sha256":sha(path),
                       "case_config_sha256":sha(BASE/name/"case_config.json"),
                       "experimental_comparison":"BLOCKED"})
    harmonic_cases=[]
    for name in ("r02_baseline","r02_refined","r04"):
        for i,cycle in enumerate(corrected["cases"][name]["cycles"]):
            harmonic_cases.append({"case":name,"cycle_index":i+1,"rate_phase_N_nm":cycle["N"]["rate_1"],
                 "acceleration_phase_N_nm":cycle["N"]["accel_1"],"third_rate_N_nm":cycle["N"]["rate_3"],
                 "third_accel_N_nm":cycle["N"]["accel_3"],"moment_origin":"moving_case_CG",
                 "frequency_hz":.08,"frequency_status":"inferred",
                 "source_file":str(harmonic_source),"source_sha256":sha(harmonic_source),
                 "experimental_comparison":"BLOCKED"})
    write("remapped_existing_results.json", {"status":"PARTIAL_MAPPING_ONLY",
          "steady_mrf":steady,"harmonic_urans":harmonic_cases,
          "notes":["MRF rows are never compared to PMM harmonics",
                   "Harmonic rows preserve distinct rate, acceleration and third terms",
                   "No experimental absolute harmonic reconstructed"]})
    print(OUT)


if __name__ == "__main__":
    main()
