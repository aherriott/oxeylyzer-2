    fn dist(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> f64 {
        let (dx, dy) = dx_dy(k1, k2, f1, f2);

        dx.hypot(dy)
    }

    fn dx_dy(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> (f64, f64) {
        let flen = |f: Finger| match f {
            Finger::LP | Finger::RP => -0.15,
            Finger::LR | Finger::RR => 0.35,
            Finger::LM | Finger::RM => 0.25,
            Finger::LI | Finger::RI => -0.30,
            Finger::LT | Finger::RT => -1.80,
        };

        let ox1 = (k1.width() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);
        let ox2 = (k1.width() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);

        let oy1 = (k2.height() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);
        let oy2 = (k2.height() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);

        let l1 = k1.x() + ox1;
        let r1 = k1.x() - ox1 + k1.width();
        let t1 = k1.y() + oy1 + flen(f1);
        let b1 = k1.y() - oy1 + k1.height() + flen(f1);

        let l2 = k2.x() + ox2;
        let r2 = k2.x() - ox2 + k2.width();
        let t2 = k2.y() + oy2 + flen(f2);
        let b2 = k2.y() - oy2 + k2.height() + flen(f2);

        let dx = (l1.max(l2) - r1.min(r2)).max(0.0);
        let dy = (t1.max(t2) - b1.min(b2)).max(0.0);

        // Checks whether or not a finger is below or to the side of another finger, in which case the
        // distance is considered negative. To the side meaning, where the distance between qwerty `er`
        // pressed with middle and index is considered 1, if each key were pressed with the other
        // finger, the distance is negative (because who the fuck is doing that, that's not good).

        let xo = Self::x_finger_overlap(f1, f2);

        // match (f1.hand(), f2.hand()) {
        //     (Hand::Left, Hand::Left) => match ((f1 as CacheKey) > (f2 as CacheKey), (f1 as CacheKey) < (f2 as CacheKey)) {
        //         (true, false) if r1 < l2 => (-dx, dy),
        //         (false, true) if l1 > r2 => (-dx, dy),
        //         _ => (dx, dy),
        //     },
        //     (Hand::Right, Hand::Right) => match ((f2 as CacheKey) > (f1 as CacheKey), (f2 as CacheKey) < (f1 as CacheKey)) {
        //         (true, false) if r1 > l2 => (-dx, dy),
        //         (false, true) if l1 < r2 => (-dx, dy),
        //         _ => (dx, dy),
        //     },
        //     _ => (dx, dy)
        // }
        match (
            (f1 as CacheKey) > (f2 as CacheKey),
            (f1 as CacheKey) < (f2 as CacheKey),
        ) {
            (true, false) if r1 < l2 + xo => (-dx, dy),
            (false, true) if l1 + xo > r2 => (-dx, dy),
            _ => (dx, dy),
        }
    }

        fn x_overlap(dx: f64, dy: f64, f1: Finger, f2: Finger) -> f64 {
        let x_offset = Self::x_finger_overlap(f1, f2);

        let dx_offset = x_offset - dx * 1.3;
        let dy_offset = 0.3333 * dy;

        (dx_offset + dy_offset).max(0.0)
    }

        fn x_finger_overlap(f1: Finger, f2: Finger) -> f64 {
        match (f1, f2) {
            (Finger::LP, Finger::LR) => 0.8,
            (Finger::LR, Finger::LP) => 0.8,
            (Finger::LR, Finger::LM) => 0.4,
            (Finger::LM, Finger::LR) => 0.4,
            (Finger::LM, Finger::LI) => 0.1,
            (Finger::LI, Finger::LM) => 0.1,
            (Finger::LI, Finger::LT) => -2.5,
            (Finger::LT, Finger::LI) => -2.5,
            (Finger::RT, Finger::RI) => -2.5,
            (Finger::RI, Finger::RT) => -2.5,
            (Finger::RI, Finger::RM) => 0.1,
            (Finger::RM, Finger::RI) => 0.1,
            (Finger::RM, Finger::RR) => 0.4,
            (Finger::RR, Finger::RM) => 0.4,
            (Finger::RR, Finger::RP) => 0.8,
            (Finger::RP, Finger::RR) => 0.8,
            _ => 0.0,
        }
    }

