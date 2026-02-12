-- Seed departments
INSERT INTO departments (id, name) VALUES
  ('dept_custom_builds', 'Custom Builds'),
  ('dept_repairs', 'Repairs'),
  ('dept_retail', 'Retail & Merch');

-- Seed employees
INSERT INTO employees (id, name, role, department_id, manager_id) VALUES
  ('person_peter_tamisin', 'Peter "Lil Dro" Tamisin', 'CEO', NULL, NULL),
  ('person_rosa_martinez', 'Rosa Martinez', 'Director of Custom Builds', 'dept_custom_builds', 'person_peter_tamisin'),
  ('person_jonas_reed', 'Jonas Reed', 'Director of Repairs', 'dept_repairs', 'person_peter_tamisin'),
  ('person_samira_patel', 'Samira Patel', 'Director of Retail & Merch', 'dept_retail', 'person_peter_tamisin'),

  ('person_derek_vaughn', 'Derek Vaughn', 'Senior Fabricator', 'dept_custom_builds', 'person_rosa_martinez'),
  ('person_lina_chavez', 'Lina Chavez', 'Frame Builder', 'dept_custom_builds', 'person_rosa_martinez'),

  ('person_marco_diaz', 'Marco Diaz', 'Lead Mechanic', 'dept_repairs', 'person_jonas_reed'),
  ('person_priya_sharma', 'Priya Sharma', 'Mechanic', 'dept_repairs', 'person_jonas_reed'),
  ('person_ron_banks', 'Ron Banks', 'Mechanic', 'dept_repairs', 'person_jonas_reed'),

  ('person_alyssa_kent', 'Alyssa Kent', 'Sales Associate', 'dept_retail', 'person_samira_patel'),
  ('person_henry_cho', 'Henry Cho', 'Sales Associate', 'dept_retail', 'person_samira_patel'),
  ('person_talia_boyd', 'Talia Boyd', 'Customer Support', 'dept_retail', 'person_samira_patel'),
  ('person_jordan_fields', 'Jordan Fields', 'Customer Support', 'dept_retail', 'person_samira_patel');

-- Seed Q3 2024 sales metrics
INSERT INTO sales_metrics (id, person_id, year, month, amount_usd) VALUES
  -- Derek Vaughn
  ('sales_person_derek_vaughn_2024-07', 'person_derek_vaughn', 2024, '07', 4200),
  ('sales_person_derek_vaughn_2024-08', 'person_derek_vaughn', 2024, '08', 4900),
  ('sales_person_derek_vaughn_2024-09', 'person_derek_vaughn', 2024, '09', 5300),

  -- Lina Chavez
  ('sales_person_lina_chavez_2024-07', 'person_lina_chavez', 2024, '07', 3800),
  ('sales_person_lina_chavez_2024-08', 'person_lina_chavez', 2024, '08', 4100),
  ('sales_person_lina_chavez_2024-09', 'person_lina_chavez', 2024, '09', 4500),

  -- Marco Diaz
  ('sales_person_marco_diaz_2024-07', 'person_marco_diaz', 2024, '07', 3200),
  ('sales_person_marco_diaz_2024-08', 'person_marco_diaz', 2024, '08', 3400),
  ('sales_person_marco_diaz_2024-09', 'person_marco_diaz', 2024, '09', 3900),

  -- Priya Sharma
  ('sales_person_priya_sharma_2024-07', 'person_priya_sharma', 2024, '07', 2900),
  ('sales_person_priya_sharma_2024-08', 'person_priya_sharma', 2024, '08', 3200),
  ('sales_person_priya_sharma_2024-09', 'person_priya_sharma', 2024, '09', 3600),

  -- Ron Banks
  ('sales_person_ron_banks_2024-07', 'person_ron_banks', 2024, '07', 2400),
  ('sales_person_ron_banks_2024-08', 'person_ron_banks', 2024, '08', 2700),
  ('sales_person_ron_banks_2024-09', 'person_ron_banks', 2024, '09', 3000),

  -- Alyssa Kent
  ('sales_person_alyssa_kent_2024-07', 'person_alyssa_kent', 2024, '07', 3000),
  ('sales_person_alyssa_kent_2024-08', 'person_alyssa_kent', 2024, '08', 3300),
  ('sales_person_alyssa_kent_2024-09', 'person_alyssa_kent', 2024, '09', 3800),

  -- Henry Cho
  ('sales_person_henry_cho_2024-07', 'person_henry_cho', 2024, '07', 2700),
  ('sales_person_henry_cho_2024-08', 'person_henry_cho', 2024, '08', 3000),
  ('sales_person_henry_cho_2024-09', 'person_henry_cho', 2024, '09', 3200),

  -- Talia Boyd
  ('sales_person_talia_boyd_2024-07', 'person_talia_boyd', 2024, '07', 1900),
  ('sales_person_talia_boyd_2024-08', 'person_talia_boyd', 2024, '08', 2100),
  ('sales_person_talia_boyd_2024-09', 'person_talia_boyd', 2024, '09', 2300),

  -- Jordan Fields
  ('sales_person_jordan_fields_2024-07', 'person_jordan_fields', 2024, '07', 1800),
  ('sales_person_jordan_fields_2024-08', 'person_jordan_fields', 2024, '08', 2000),
  ('sales_person_jordan_fields_2024-09', 'person_jordan_fields', 2024, '09', 2200);

-- Products (from Product_* and repair/custom build docs)
INSERT INTO products (id, name, category) VALUES
  ('prod_custom_frame_build', 'Custom Frame Build', 'custom_build'),
  ('prod_wheel_tire_set', 'Wheel & Tire Set', 'parts'),
  ('prod_brake_service', 'Brake Service', 'labor'),
  ('prod_drivetrain_gears', 'Drivetrain / Gears', 'parts'),
  ('prod_paint_finish', 'Paint & Finish', 'custom_build'),
  ('prod_customization_accessories', 'Customization & Accessories', 'accessories'),
  ('prod_parts', 'Parts (rims, grips, forks)', 'parts'),
  ('prod_repair_labor', 'Repair Labor', 'labor'),
  ('prod_tuneup', 'Tune-up Service', 'labor');

-- Marketing campaigns (from Marketing_* PDFs)
INSERT INTO marketing_campaigns (id, name, description) VALUES
  ('camp_holiday_promo', 'Holiday Promo', '20% off parts, 15% off labor, $100 build deposits'),
  ('camp_spring_ride', 'Spring Community Ride', '$25 tune-ups, 10% accessory discounts'),
  ('camp_build_showcase', 'Custom Build Showcase', 'Quarterly featured custom builds'),
  ('camp_social_promo', 'Social Media Promo', 'Promotional content, influencer partnerships'),
  ('camp_winter_warmer', 'Winter Warmer Promotion', 'Seasonal promotion');

-- Sales transactions (roll up to sales_metrics totals; product + optional campaign)
INSERT INTO sales_transactions (id, transaction_date, person_id, product_id, amount_usd, marketing_campaign_id, year, month) VALUES
  -- Derek Vaughn Jul 4200
  ('tx_001', '2024-07-02', 'person_derek_vaughn', 'prod_custom_frame_build', 1800, 'camp_build_showcase', 2024, '07'),
  ('tx_002', '2024-07-08', 'person_derek_vaughn', 'prod_repair_labor', 950, NULL, 2024, '07'),
  ('tx_003', '2024-07-14', 'person_derek_vaughn', 'prod_parts', 650, NULL, 2024, '07'),
  ('tx_004', '2024-07-22', 'person_derek_vaughn', 'prod_paint_finish', 800, NULL, 2024, '07'),
  -- Derek Vaughn Aug 4900
  ('tx_005', '2024-08-01', 'person_derek_vaughn', 'prod_custom_frame_build', 2200, 'camp_social_promo', 2024, '08'),
  ('tx_006', '2024-08-10', 'person_derek_vaughn', 'prod_wheel_tire_set', 900, NULL, 2024, '08'),
  ('tx_007', '2024-08-18', 'person_derek_vaughn', 'prod_repair_labor', 1100, NULL, 2024, '08'),
  ('tx_008', '2024-08-25', 'person_derek_vaughn', 'prod_parts', 700, NULL, 2024, '08'),
  -- Derek Vaughn Sep 5300
  ('tx_009', '2024-09-05', 'person_derek_vaughn', 'prod_custom_frame_build', 2400, 'camp_build_showcase', 2024, '09'),
  ('tx_010', '2024-09-12', 'person_derek_vaughn', 'prod_paint_finish', 1200, NULL, 2024, '09'),
  ('tx_011', '2024-09-20', 'person_derek_vaughn', 'prod_repair_labor', 1000, NULL, 2024, '09'),
  ('tx_012', '2024-09-28', 'person_derek_vaughn', 'prod_parts', 700, NULL, 2024, '09'),
  -- Lina Chavez Jul 3800
  ('tx_013', '2024-07-03', 'person_lina_chavez', 'prod_custom_frame_build', 1500, NULL, 2024, '07'),
  ('tx_014', '2024-07-11', 'person_lina_chavez', 'prod_paint_finish', 800, 'camp_build_showcase', 2024, '07'),
  ('tx_015', '2024-07-19', 'person_lina_chavez', 'prod_repair_labor', 900, NULL, 2024, '07'),
  ('tx_016', '2024-07-26', 'person_lina_chavez', 'prod_parts', 600, NULL, 2024, '07'),
  -- Lina Chavez Aug 4100
  ('tx_017', '2024-08-04', 'person_lina_chavez', 'prod_custom_frame_build', 1900, NULL, 2024, '08'),
  ('tx_018', '2024-08-12', 'person_lina_chavez', 'prod_wheel_tire_set', 750, NULL, 2024, '08'),
  ('tx_019', '2024-08-20', 'person_lina_chavez', 'prod_repair_labor', 950, 'camp_spring_ride', 2024, '08'),
  ('tx_020', '2024-08-27', 'person_lina_chavez', 'prod_parts', 500, NULL, 2024, '08'),
  -- Lina Chavez Sep 4500
  ('tx_021', '2024-09-06', 'person_lina_chavez', 'prod_custom_frame_build', 2000, 'camp_build_showcase', 2024, '09'),
  ('tx_022', '2024-09-14', 'person_lina_chavez', 'prod_paint_finish', 1000, NULL, 2024, '09'),
  ('tx_023', '2024-09-22', 'person_lina_chavez', 'prod_repair_labor', 850, NULL, 2024, '09'),
  ('tx_024', '2024-09-29', 'person_lina_chavez', 'prod_parts', 650, NULL, 2024, '09'),
  -- Marco Diaz Jul 3200
  ('tx_025', '2024-07-05', 'person_marco_diaz', 'prod_repair_labor', 1200, NULL, 2024, '07'),
  ('tx_026', '2024-07-13', 'person_marco_diaz', 'prod_brake_service', 800, NULL, 2024, '07'),
  ('tx_027', '2024-07-21', 'person_marco_diaz', 'prod_parts', 600, 'camp_social_promo', 2024, '07'),
  ('tx_028', '2024-07-28', 'person_marco_diaz', 'prod_tuneup', 600, NULL, 2024, '07'),
  -- Marco Diaz Aug 3400
  ('tx_029', '2024-08-06', 'person_marco_diaz', 'prod_repair_labor', 1100, NULL, 2024, '08'),
  ('tx_030', '2024-08-14', 'person_marco_diaz', 'prod_drivetrain_gears', 900, NULL, 2024, '08'),
  ('tx_031', '2024-08-22', 'person_marco_diaz', 'prod_parts', 700, NULL, 2024, '08'),
  ('tx_032', '2024-08-29', 'person_marco_diaz', 'prod_tuneup', 700, 'camp_spring_ride', 2024, '08'),
  -- Marco Diaz Sep 3900
  ('tx_033', '2024-09-07', 'person_marco_diaz', 'prod_repair_labor', 1400, NULL, 2024, '09'),
  ('tx_034', '2024-09-15', 'person_marco_diaz', 'prod_brake_service', 1000, NULL, 2024, '09'),
  ('tx_035', '2024-09-23', 'person_marco_diaz', 'prod_parts', 800, NULL, 2024, '09'),
  ('tx_036', '2024-09-30', 'person_marco_diaz', 'prod_tuneup', 700, NULL, 2024, '09'),
  -- Priya Sharma Jul 2900
  ('tx_037', '2024-07-04', 'person_priya_sharma', 'prod_repair_labor', 1000, NULL, 2024, '07'),
  ('tx_038', '2024-07-15', 'person_priya_sharma', 'prod_brake_service', 700, NULL, 2024, '07'),
  ('tx_039', '2024-07-24', 'person_priya_sharma', 'prod_parts', 500, NULL, 2024, '07'),
  ('tx_040', '2024-07-30', 'person_priya_sharma', 'prod_tuneup', 700, NULL, 2024, '07'),
  -- Priya Sharma Aug 3200
  ('tx_041', '2024-08-05', 'person_priya_sharma', 'prod_repair_labor', 1100, NULL, 2024, '08'),
  ('tx_042', '2024-08-16', 'person_priya_sharma', 'prod_drivetrain_gears', 800, 'camp_social_promo', 2024, '08'),
  ('tx_043', '2024-08-24', 'person_priya_sharma', 'prod_parts', 600, NULL, 2024, '08'),
  ('tx_044', '2024-08-31', 'person_priya_sharma', 'prod_tuneup', 700, NULL, 2024, '08'),
  -- Priya Sharma Sep 3600
  ('tx_045', '2024-09-08', 'person_priya_sharma', 'prod_repair_labor', 1300, NULL, 2024, '09'),
  ('tx_046', '2024-09-16', 'person_priya_sharma', 'prod_brake_service', 900, NULL, 2024, '09'),
  ('tx_047', '2024-09-25', 'person_priya_sharma', 'prod_parts', 700, NULL, 2024, '09'),
  ('tx_048', '2024-09-30', 'person_priya_sharma', 'prod_tuneup', 700, NULL, 2024, '09'),
  -- Ron Banks Jul 2400
  ('tx_049', '2024-07-06', 'person_ron_banks', 'prod_repair_labor', 900, NULL, 2024, '07'),
  ('tx_050', '2024-07-17', 'person_ron_banks', 'prod_tuneup', 500, NULL, 2024, '07'),
  ('tx_051', '2024-07-25', 'person_ron_banks', 'prod_parts', 500, NULL, 2024, '07'),
  ('tx_052', '2024-07-31', 'person_ron_banks', 'prod_brake_service', 500, NULL, 2024, '07'),
  -- Ron Banks Aug 2700
  ('tx_053', '2024-08-07', 'person_ron_banks', 'prod_repair_labor', 1000, NULL, 2024, '08'),
  ('tx_054', '2024-08-18', 'person_ron_banks', 'prod_tuneup', 600, 'camp_spring_ride', 2024, '08'),
  ('tx_055', '2024-08-26', 'person_ron_banks', 'prod_parts', 550, NULL, 2024, '08'),
  ('tx_056', '2024-08-31', 'person_ron_banks', 'prod_brake_service', 550, NULL, 2024, '08'),
  -- Ron Banks Sep 3000
  ('tx_057', '2024-09-09', 'person_ron_banks', 'prod_repair_labor', 1100, NULL, 2024, '09'),
  ('tx_058', '2024-09-17', 'person_ron_banks', 'prod_tuneup', 700, NULL, 2024, '09'),
  ('tx_059', '2024-09-26', 'person_ron_banks', 'prod_parts', 600, NULL, 2024, '09'),
  ('tx_060', '2024-09-30', 'person_ron_banks', 'prod_brake_service', 600, NULL, 2024, '09'),
  -- Alyssa Kent Jul 3000
  ('tx_061', '2024-07-07', 'person_alyssa_kent', 'prod_customization_accessories', 900, NULL, 2024, '07'),
  ('tx_062', '2024-07-16', 'person_alyssa_kent', 'prod_wheel_tire_set', 800, 'camp_build_showcase', 2024, '07'),
  ('tx_063', '2024-07-24', 'person_alyssa_kent', 'prod_parts', 600, NULL, 2024, '07'),
  ('tx_064', '2024-07-29', 'person_alyssa_kent', 'prod_tuneup', 700, NULL, 2024, '07'),
  -- Alyssa Kent Aug 3300
  ('tx_065', '2024-08-08', 'person_alyssa_kent', 'prod_customization_accessories', 1000, NULL, 2024, '08'),
  ('tx_066', '2024-08-19', 'person_alyssa_kent', 'prod_parts', 800, NULL, 2024, '08'),
  ('tx_067', '2024-08-25', 'person_alyssa_kent', 'prod_wheel_tire_set', 750, NULL, 2024, '08'),
  ('tx_068', '2024-08-30', 'person_alyssa_kent', 'prod_tuneup', 750, 'camp_social_promo', 2024, '08'),
  -- Alyssa Kent Sep 3800
  ('tx_069', '2024-09-10', 'person_alyssa_kent', 'prod_customization_accessories', 1200, NULL, 2024, '09'),
  ('tx_070', '2024-09-18', 'person_alyssa_kent', 'prod_wheel_tire_set', 1000, NULL, 2024, '09'),
  ('tx_071', '2024-09-24', 'person_alyssa_kent', 'prod_parts', 800, NULL, 2024, '09'),
  ('tx_072', '2024-09-29', 'person_alyssa_kent', 'prod_tuneup', 800, NULL, 2024, '09'),
  -- Henry Cho Jul 2700
  ('tx_073', '2024-07-09', 'person_henry_cho', 'prod_parts', 700, NULL, 2024, '07'),
  ('tx_074', '2024-07-18', 'person_henry_cho', 'prod_customization_accessories', 800, NULL, 2024, '07'),
  ('tx_075', '2024-07-27', 'person_henry_cho', 'prod_tuneup', 600, 'camp_spring_ride', 2024, '07'),
  ('tx_076', '2024-07-31', 'person_henry_cho', 'prod_wheel_tire_set', 600, NULL, 2024, '07'),
  -- Henry Cho Aug 3000
  ('tx_077', '2024-08-09', 'person_henry_cho', 'prod_parts', 800, NULL, 2024, '08'),
  ('tx_078', '2024-08-20', 'person_henry_cho', 'prod_customization_accessories', 900, NULL, 2024, '08'),
  ('tx_079', '2024-08-28', 'person_henry_cho', 'prod_tuneup', 650, NULL, 2024, '08'),
  ('tx_080', '2024-08-31', 'person_henry_cho', 'prod_wheel_tire_set', 650, NULL, 2024, '08'),
  -- Henry Cho Sep 3200
  ('tx_081', '2024-09-11', 'person_henry_cho', 'prod_parts', 900, NULL, 2024, '09'),
  ('tx_082', '2024-09-19', 'person_henry_cho', 'prod_customization_accessories', 950, 'camp_social_promo', 2024, '09'),
  ('tx_083', '2024-09-27', 'person_henry_cho', 'prod_tuneup', 700, NULL, 2024, '09'),
  ('tx_084', '2024-09-30', 'person_henry_cho', 'prod_wheel_tire_set', 650, NULL, 2024, '09'),
  -- Talia Boyd Jul 1900
  ('tx_085', '2024-07-10', 'person_talia_boyd', 'prod_parts', 500, NULL, 2024, '07'),
  ('tx_086', '2024-07-20', 'person_talia_boyd', 'prod_customization_accessories', 600, NULL, 2024, '07'),
  ('tx_087', '2024-07-28', 'person_talia_boyd', 'prod_tuneup', 800, NULL, 2024, '07'),
  -- Talia Boyd Aug 2100
  ('tx_088', '2024-08-11', 'person_talia_boyd', 'prod_parts', 600, NULL, 2024, '08'),
  ('tx_089', '2024-08-21', 'person_talia_boyd', 'prod_customization_accessories', 650, 'camp_social_promo', 2024, '08'),
  ('tx_090', '2024-08-30', 'person_talia_boyd', 'prod_tuneup', 850, NULL, 2024, '08'),
  -- Talia Boyd Sep 2300
  ('tx_091', '2024-09-12', 'person_talia_boyd', 'prod_parts', 600, NULL, 2024, '09'),
  ('tx_092', '2024-09-21', 'person_talia_boyd', 'prod_customization_accessories', 700, NULL, 2024, '09'),
  ('tx_093', '2024-09-29', 'person_talia_boyd', 'prod_tuneup', 1000, NULL, 2024, '09'),
  -- Jordan Fields Jul 1800
  ('tx_094', '2024-07-11', 'person_jordan_fields', 'prod_parts', 500, NULL, 2024, '07'),
  ('tx_095', '2024-07-22', 'person_jordan_fields', 'prod_tuneup', 600, NULL, 2024, '07'),
  ('tx_096', '2024-07-30', 'person_jordan_fields', 'prod_customization_accessories', 700, NULL, 2024, '07'),
  -- Jordan Fields Aug 2000
  ('tx_097', '2024-08-12', 'person_jordan_fields', 'prod_parts', 550, NULL, 2024, '08'),
  ('tx_098', '2024-08-23', 'person_jordan_fields', 'prod_tuneup', 700, NULL, 2024, '08'),
  ('tx_099', '2024-08-31', 'person_jordan_fields', 'prod_customization_accessories', 750, 'camp_spring_ride', 2024, '08'),
  -- Jordan Fields Sep 2200
  ('tx_100', '2024-09-13', 'person_jordan_fields', 'prod_parts', 600, NULL, 2024, '09'),
  ('tx_101', '2024-09-22', 'person_jordan_fields', 'prod_tuneup', 800, NULL, 2024, '09'),
  ('tx_102', '2024-09-30', 'person_jordan_fields', 'prod_customization_accessories', 800, NULL, 2024, '09');

