use libdof::Dof;
use oxeylyzer_core::{
    prelude::{Analyzer, Data},
    weights::dummy_weights,
};

use std::{collections::HashMap, fs, io::Write};

fn main() {
    let analyzers = fs::read_dir("./data")
        .unwrap()
        .flat_map(|r| r.inspect_err(|e| println!("couldn't create direntry for data: {e}")))
        .flat_map(|d| {
            Data::load(d.path())
                .inspect_err(|e| println!("couldn't read string from file for data: {e}"))
        })
        .map(|d| (d.name.to_lowercase(), Analyzer::new(d, dummy_weights())))
        .collect::<HashMap<_, _>>();

    let lines = fs::read_dir("./data-csv/dofs/")
        .unwrap()
        .flat_map(|r| r.inspect_err(|e| println!("couldn't create direntry for dofs: {e}")))
        .flat_map(|d| {
            fs::read_to_string(d.path())
                .inspect_err(|e| println!("couldn't read string from file for dofs: {e}"))
        })
        .flat_map(|s| {
            serde_json::from_str::<Dof>(&s).inspect_err(|e| println!("couldn't parse dof! {e}"))
        })
        .map(|dof| {
            let data_lang = format!("{}_no_space", dof.languages()[0].language.to_lowercase());
            let name = dof.name().to_owned();
            let lang = dof.languages()[0].language.to_owned();
            let s = analyzers.get(&data_lang).unwrap().stats(&dof.into());
            let t = s.trigrams;

            format!(
                "{name},{lang},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                s.sfbs,
                s.sfs,
                t.sft,
                t.alternate,
                t.inroll,
                t.outroll,
                t.redirect,
                t.onehandin,
                t.onehandout,
                t.invalid,
                s.finger_sfbs.map(|f| f.to_string()).join(","),
                s.finger_use.map(|f| f.to_string()).join(","),
                s.weighted_finger_distance.map(|f| f.to_string()).join(","),
                s.unweighted_finger_distance
                    .map(|f| f.to_string())
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut f = fs::File::options()
        .create(true)
        .truncate(true)
        .write(true)
        .open("./data-csv/cmini-layouts-data.csv")
        .unwrap();

    f.write_all(
        concat!(
            "name,language,sfbs,sfs,sft,alternate,inroll,outroll,redirect,onehandin,onehandout,invalid,",
            "sfbs_LP,sfbs_LR,sfbs_LM,sfbs_LI,sfbs_LT,sfbs_RT,sfbs_RI,sfbs_RM,sfbs_RR,sfbs_RP,",
            "use_LP,use_LR,use_LM,use_LI,use_LT,use_RT,use_RI,use_RM,use_RR,use_RP,",
            "weighted_d_LP,weighted_d_LR,weighted_d_LM,weighted_d_LI,weighted_d_LT,",
            "weighted_d_RT,weighted_d_RI,weighted_d_RM,weighted_d_RR,weighted_d_RP",
            "unweighted_d_LP,unweighted_d_LR,unweighted_d_LM,unweighted_d_LI,unweighted_d_LT,",
            "unweighted_d_RT,unweighted_d_RI,unweighted_d_RM,unweighted_d_RR,unweighted_d_RP\n",
        ).as_bytes()
    ).unwrap();

    f.write_all(lines.as_bytes()).unwrap();
}
