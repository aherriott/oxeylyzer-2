#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::cached_layout::CachedLayout;
    use crate::weights::dummy_weights;

    #[test]
    fn count_trigram_combos() {
        let data = crate::data::Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");
        let cached = CachedLayout::new(&layout, data, &weights);

        // We need to access trigram combo counts. Let's add a method.
        let (first, mid, end) = cached.trigram_combo_counts();
        println!("\n=== Trigram combo counts ===");
        println!("Per-position first: {:?}", first);
        println!("Per-position mid: {:?}", mid);
        println!("Per-position end: {:?}", end);
        println!("Total first: {}", first.iter().sum::<usize>());
        println!("Total mid: {}", mid.iter().sum::<usize>());
        println!("Total end: {}", end.iter().sum::<usize>());
    }
}
