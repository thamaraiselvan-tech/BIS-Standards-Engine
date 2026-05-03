from __future__ import annotations
"""
rag_pipeline.py — BIS Standards Recommendation Engine
Complete knowledge base: 535 standards from SP 21 (2005) — all 27 sections, 929 pages

Sections covered:
  1  Cement and Concrete          |  2  Building Limes
  3  Stones                       |  4  Clay Products for Building
  5  Gypsum Building Materials    |  6  Timber
  7  Bitumen and Tar Products     |  8  Floor, Wall, Roof Coverings
  9  Waterproofing                | 10  Sanitary Appliances
  11 Builders Hardware            | 12  Wood Products
  13 Doors, Windows and Shutters  | 14  Concrete Reinforcement
  15 Structural Steels            | 16  Light Metals and Alloys
  17 Structural Shapes            | 18  Welding Electrodes
  19 Threaded Fasteners           | 20  Wire Ropes
  21 Glass                        | 22  Fillers and Putties
  23 Thermal Insulation           | 24  Plastics
  25 Conductors and Cables        | 26  Wiring Accessories
  27 General

Architecture:
  Query Expansion (80+ rules) → BM25 (40%) + TF-IDF trigram (60%) → Top-5
  LLM called ONLY for explanation streaming — never in retrieval path
"""

import re, time, json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

STANDARDS_DB = {
    "IS 383: 1970": {
        "title":    "COARSE AND FINE AGGREGATES FROM NATURAL SOURCES FOR CONCRETE 1. Scope \u2014 Requirements for aggregates, crushed or uncrushed, derived from natural source",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements for aggregates, crushed or uncrushed, derived from natural sources for use in the production of structural concrete including mass concrete works.",
    },
    "IS 2116: 1980": {
        "title":    "Sand for Masonry Mortars",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements of naturally occurring sands, crushed stone sands and crushed gravel sands used in mortars for construction of masonry.",
    },
    "IS 9142: 1979": {
        "title":    "ARTIFICIAL LIGHTWEIGHT AGGREGATES FOR CONCRETE MASONRY UNITS 1. Scope \u2014 Requirements of artificial lightweight aggregates, such as foamed blast furnac",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements of artificial lightweight aggregates, such as foamed blast furnace slag, bloated clay aggregate, sintered fly ash aggregate and cinder aggregate intended for use in concrete masonry units in which prime consideration is lightness in mass.",
    },
    "IS 269: 1989": {
        "title":    "ORDINARY PORTLAND CEMENT, 33 GRADE 1. Scope \u2014 Covers the manufacture and chemical and physical requirements of 33 grade ordinary Portland cement. 2. C",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Covers the manufacture and chemical and physical requirements of 33 grade ordinary Portland cement.",
    },
    "IS 455: 1989": {
        "title":    "PORTLAND SLAG CEMENT (Fourth Revison) Note \u2014 For methods of tests, refer to relevant parts of IS 4031 Methods of physical tests for hydraulic cement a",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Covers the manufacture and chemical and physical requirements for Portland slag cement.",
    },
    "IS 1489 (PART 1): 1991": {
        "title":    "PORTLAND POZZOLANA CEMENT PART 1 FLY ASH BASED 1. Scope \u2014 Covers the manufacture, physical and chemical requirements of Portland pozzolana cement usin",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Covers the manufacture, physical and chemical requirements of Portland pozzolana cement using only fly ash pozzolana.",
    },
    "IS 1489 (PART 2): 1991": {
        "title":    "PORTLAND POZZOLANA CEMENT PART 2 CALCINED CLAY BASED TABLE 1 CHEMICAL REQUIREMENTS OF PORTLAND- POZZOLANA CEMENT SI No. Characteristic Requirement (1)",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Manufacture, Physical and Chemical requriements of Portland- pozzolana cement manufactured by using calcined clay pozzolana or a mixture of calcined clay and fly ash pozzolana.",
    },
    "IS 3466: 1988": {
        "title":    "Masonry Cement",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements for masonry cement to be used for all general purposes where mortars for masonry are required. Masonry cement is not intended for use in structural concrete, for flooring and foundation work or for reinforced and prestressed concrete works.",
    },
    "IS 6452: 1989": {
        "title":    "HIGH ALUMINA CEMENT FOR STRUCTURAL USE Fineness \u2014 Specific surface not less than 225 m2/kg Soundness \u2014 Expansion not more than 5 mm (quantity of mixin",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "2.2 Fineness \u2014 Specific surface not less than 225 m2/kg 2.3 Soundness \u2014 Expansion not more than 5 mm (quantity of mixing water shall be 22 percent of cement by mass). 2.4 Setting Time \u2014 Initial not less than 30 minutes",
    },
    "IS 6909: 1990": {
        "title":    "SUPERSULPHATED CEMENT Note \u2014 For methods of tests, refer to relevant parts of IS 4031 Methods of physical tests for hydraulic cement, and Method of ch",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Note \u2014 For methods of tests, refer to relevant parts of IS 4031 Methods of physical tests for hydraulic cement, and IS 4032:1985 Method of chemical analysis of hydraulic cement (first revision) For detailed information, refer to IS 6909:1990 Specification for supersulphated cement (first revision).",
    },
    "IS 8041: 1990": {
        "title":    "RAPID HARDENING PORTLAND CEMENT Not more than 10 mm ('Le Chatelier\u2019 method). Not more than percent (autoclave). Setting Time: Initial setting 30 minut",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Not more than 10 mm ('Le Chatelier\u2019 method). Not more than 0.8 percent (autoclave). 3.3 Setting Time: Initial setting 30 minutes, final setting 10 h. 3.4 Compressive Strength of Mortar Cubes a)",
    },
    "IS 8042: 1989": {
        "title":    "WHITE PORTLAND CEMENT prepared from white portland cement shall not be less than 90 percent of those specified for 33 grade ordinary Portland cement.",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "prepared from white portland cement shall not be less than 90 percent of those specified for 33 grade ordinary Portland cement. 4. Degree of Whiteness \u2014 The reflectance of neat cement ring prepared and tested in accordance with the test specified shall not be less than 70 percent.",
    },
    "IS 8043: 1991": {
        "title":    "HYDROPHOBIC PORTLAND CEMENT 3. Physical Requirements 3.1Fineness \u2014 Specific surface shall not be less than 350 m2/kg. 3.2Soundness and Setting Time \u2014",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "3. Physical Requirements 3.1Fineness \u2014 Specific surface shall not be less than 350 m2/kg. 3.2Soundness and Setting Time \u2014 Shall be as laid down IS 269:1989. 3.3Compressive Strength",
    },
    "IS 8112: 1989": {
        "title":    "43 GRADE ORDINARY PORTLAND CEMENT Setting Time \u2014 a) Initial setting time in minutes \u2014not less than 30. b) Final setting time in minutes \u2014 not more tha",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "3.3 Setting Time \u2014 a) Initial setting time in minutes \u2014not less than 30. b) Final setting time in minutes \u2014 not more than 600. 3.4 Compressive strength \u2014",
    },
    "IS 12269: 1987": {
        "title":    "53 GRADE ORDINARY PORTLAND CEMENT Soundness \u2014 unaerated cement not more than 10 mm by \u2018Le Chatelier\u2019 method and percent by autoclave method Setting Ti",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "mm by \u2018Le Chatelier\u2019 method and 0.8 percent by autoclave method 3.3 Setting Time \u2014 a) Initial setting time in minutes \u2013 not less than 30, and b)",
    },
    "IS 12330: 1988": {
        "title":    "SULPHATE RESISTING PORTLAND CEMENT 3. Physical Requirement Fineness \u2014 Specific surface not less than 225 m2/kg Soundness \u2013 Unaerated cement-expansion",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "3.1 Fineness \u2014 Specific surface not less than 225 m2/kg 3.2 Soundness \u2013 Unaerated cement-expansion not more than 10 mm by \u2018Le Chatelier\u2019 method and not more than 0.8 percent by autoclave method. 3.3 Setting Time \u2014",
    },
    "IS 2185 (PART 1): 1979": {
        "title":    "CONCRETE MASONRY UNITS PART 1 HOLLOW AND SOLID CONCRETE BLOCKS Note 2 \u2014 Block shall also be manufactured in half lengths of 200, 250 or 300 mm. Tolera",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "(Second Revision) Note 2 \u2014 Block shall also be manufactured in half lengths of 200, 250 or 300 mm. 3.2 Tolerances \u2014 Not more than \u00b1 5 mm in length and 3 mm in height and width of unit. 3.3 Face shells and webs shall increase in thickness",
    },
    "IS 2185 (PART 2): 1983": {
        "title":    "CONCRETE MASONRY UNITS PART 2 HOLLOW AND SOLID LIGHTWEIGHT CONCRETE BLOCKS 3. Classification Load bearing lightweight concrete masonry units hollow (o",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "CONCRETE BLOCKS (First Revision) 3. Classification 3.1 Load bearing lightweight concrete masonry units hollow (open and closed cavity) or solid shall conform to the following two grades\u2014",
    },
    "IS 2185 (PART 3): 1984": {
        "title":    "CONCRETE MASONRY UNITS PART 3 AUTOCLAVED CELLULAR (AERATED) CONCRETE BLOCKS TABLE 1 PHYSICAL PROPERTIES OF AUTOCLAVED CELLULAR CONCRETE BLOCKS Sl No.",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "CONCRETE BLOCKS (First Revision) TABLE 1 PHYSICAL PROPERTIES OF AUTOCLAVED CELLULAR CONCRETE BLOCKS Sl No. Density in Ovendry",
    },
    "IS 4996: 1984": {
        "title":    "REINFORCED CONCRETE FENCE POSTS cross-sectional dimensions and the reinforcement shall be adequate to conform to strength requirements given in 4. Not",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "cross-sectional dimensions and the reinforcement shall be adequate to conform to strength requirements given in 4. Note\u2014 Some of the common sizes and shapes for reinforced concrete fence posts with other details such as reinforcement, fencing wire spacing from ground level, spacing of line post",
    },
    "IS 5751: 1984": {
        "title":    "PRECAST CONCRETE COPING BLOCKS Note \u2014For minimum dimensions of the cross section for clip type and for flat bottomed coping, see Fig. 1 and 2 of the s",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Note \u2014For minimum dimensions of the cross section for clip type and for flat bottomed coping, see Fig. 1 and 2 of the standard. 2.2 Length \u2014 1 m or as agreed. 2.3 Tolerances \u2014 \u00b1 3 mm for cross-sectional profile and \u00b1 6 mm for length.",
    },
    "IS 5758: 1984": {
        "title":    "PRECAST CONCRETE KERBS Type of Product Dimensions (mm) Load to be Supported (N) a) Rectangular kerbs 150 \u00d7 300",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Type of Product Dimensions (mm) Load to be Supported (N) a) Rectangular kerbs 150 \u00d7 300",
    },
    "IS 5820: 1970": {
        "title":    "PRECAST CONCRETE CABLE COVERS Note 1 \u2014 L,W= Length, Width. T = Total thickness in case of flat type and thickness of flat portion excluding peak in ca",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "L,W= Length, Width. T = Total thickness in case of flat type and thickness of flat portion excluding peak in case of cover with peak. T'",
    },
    "IS 6072: 1971": {
        "title":    "AUTOCLAVED REINFORCED CELLULAR CONCRETE WALL SLABS 5.2Tolerances \u2014 For 500 mm and below, \u00b1 2 mm over 500 mm, \u00b1 5 mm. Note \u2014 For form tolerances for wa",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "5.2Tolerances \u2014 For 500 mm and below, \u00b1 2 mm over 500 mm, \u00b1 5 mm. Note \u2014 For form tolerances for wall slabs, refer to Table 1 of the standard. 6. Finish \u2014 Tongue at one side and groove on the other side. Alternatively groove on both sides for filling",
    },
    "IS 6073: 1971": {
        "title":    "AUTOCLAVED REINFORCED CELLULAR CONCRETE FLOOR AND ROOF SLABS 5.2Tolerances \u2014 For 500 mm and below, \u00b1 2 mm over 500 mm, \u00b1 5 mm. Note \u2014 For form toleran",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "5.2Tolerances \u2014 For 500 mm and below, \u00b1 2 mm over 500 mm, \u00b1 5 mm. Note \u2014 For form tolerances for wall slabs, refer to Table 1 of the standard. 6. Finish \u2014 Tongue at one side and groove on the other side. Alternatively groove on both sides for filling",
    },
    "IS 6523: 1983": {
        "title":    "PRECAST REINFORCED CONCRETE DOOR AND WINDOW FRAMES Note \u2014 For requirements in regard to manufacture (construction and finish, positioning of reinforce",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements for precast reinforced concrete door and window frames. Use of such frames is recommended to be restricted to a maximum opening width of 2.25 m.",
    },
    "IS 9893: 1981": {
        "title":    "PRECAST CONCRETE BLOCKS FOR LINTELS AND SILLS Note 1 \u2014 For details of material, refer to 3 of the standard. Note 2\u2014 For details of manufacture, or asp",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Note 1 \u2014 For details of material, refer to 3 of the standard. Note 2\u2014 For details of manufacture, or aspects such as construction, finish, mould, reinforcement, occuring etc, refer to 5 of the standard. For detailed information, refer to IS 9893:1981 Specification for precast concrete lintels and",
    },
    "IS 10388: 1982": {
        "title":    "CORRUGATED COIR, WOODWOOL, CEMENT ROOFING SHEETS Note \u2014 For methods of tests , refer to Appendices A to D of the standard. For detailed information, r",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Note \u2014 For methods of tests , refer to Appendices A to D of the standard. For detailed information, refer to I S 10388:1982 Specification for corrugated coir, woodwool, cement roofing sheets. TABLE 1 DIMENSIONS AND TOLERANCES FOR CORRUGATED COIR, WOODWOOL, CEMENT ROOFING SHEETS (All dimensions in mi",
    },
    "IS 12440: 1988": {
        "title":    "PRECAST CONCRETE STONE MASONRY BLOCKS For 200, 150 and 100 mm nominal thick walls, the blocks shall be of 300 \u00d7 200 \u00d7 150 mm, 300 \u00d7 150 \u00d7 150 mm and 3",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "blocks shall be of 300 \u00d7 200 \u00d7 150 mm, 300 \u00d7 150 \u00d7 150 mm and 300 \u00d7 100 \u00d7 150 mm nominal size respectively. 3.3 For accommodating vertical reinforcement required in earthquake resistant construction special block of half-width and with semi-circular recess in it (see Fig.1 of the standard) shall be",
    },
    "IS 12592: 2002": {
        "title":    "PRECAST CONCRETE MANHOLE COVER AND FRAME 4. Physical Requirements General \u2014 All covers and frames shall be sound and free from cracks and other defect",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements for precast steel reinforced cement concrete manhole covers and frames intended for use in sewerage and water drainage.",
    },
    "IS 13990: 1994": {
        "title":    "PRECAST REINFORCED CONCRETE PLANKS AND JOISTS FOR ROOFING AND FLOORING Tolerances \u2014 Casting tolerances on various dimensions of plank shall be as give",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "2.3 Tolerances \u2014 Casting tolerances on various dimensions of plank shall be as given below \u2014 Dimension Tolerance Length \u00b1 5 mm",
    },
    "IS 14143: 1994": {
        "title":    "PREFABRICATED BRICK PANEL AND PARTIALLY PRECAST CONCRETE JOIST FOR FLOORING AND ROOFING .1 Shape\u2014 Partially precast joist shall be rectangular in shap",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "2.2.1 Shape\u2014 Partially precast joist shall be rectangular in shape with steel stirrups kept projecting out which shall be tied with reinforcement along the joist to achieve monolithicity with concrete (see Fig.2). 2.2.2 Width\u2014 Shall be sufficient to support two successive spans of brick panels with",
    },
    "IS 14201: 1994": {
        "title":    "PRECAST REINFORCED CONCRETE CHANNEL UNITS FOR CONSTRUCTION OF FLOORS AND ROOFS Tolerances on Dimensions .1 Dimension Tolerance Length \u00b1 5 mm",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "2.3 Tolerances on Dimensions 2.3.1 Dimension Tolerance Length \u00b1 5 mm",
    },
    "IS 14241: 1995": {
        "title":    "PRECAST REINFORCED CONCRETE L\u2013PANEL FOR ROOFING Dimensions .1 Length\u2014 The maximum span of L-panels shall be restricted to 4 m. Lower lengths may be pr",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "2.2 Dimensions 2.2.1 Length\u2014 The maximum span of L-panels shall be restricted to 4 m. Lower lengths may be preferred, wherever possible, for easy handling. A minimum bearing on the gable walls shall be kept 60 mm on either side of the L-panels.",
    },
    "IS 459: 1992": {
        "title":    "CORRUGATED AND SEMI-CORRUGATED ASBESTOS CEMENT SHEETS Note 1 \u2014 For method of measurement of different dimensions of sheets, refer to 5 of the standard",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "(Third Revision) Note 1 \u2014 For method of measurement of different dimensions of sheets, refer to 5 of the standard. Note 2 \u2014 For methods of tests, refer to IS 5913: 2003 Methods of tests for absestos cement products (second revision). For detailed informatoin refer to IS 459:1992 Specification for co",
    },
    "IS 1592: 2003": {
        "title":    "ASBESTOS CEMENT PRESSURE PIPES TABLE 2 PRESSURE RELATIOSHIP Sl.No Nominal Diameters TP BP WP",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "TABLE 2 PRESSURE RELATIOSHIP Sl.No Nominal Diameters TP BP WP",
    },
    "IS 1626 (PART 1): 1994": {
        "title":    "ASBESTOS CEMENT BUILDING PIPES AND PIPE FITTINGS, GUTTERS, AND GUTTER FITTINGS AND ROOF FITTINGS PART 1 PIPES AND PIPE FITTINGS .2 Overall Length \u2014 Th",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "AND ROOF FITTINGS PART 1 PIPES AND PIPE FITTINGS (Second Revision) 3.2.2 Overall Length \u2014 The overall length is the sum of nominal length and length of socket. 3.3 Tolerances 3.3.1 Internal diameter of plain ends and sockets: The",
    },
    "IS 1626 (PART 2): 1994": {
        "title":    "ASBESTOS CEMENT BUILDING PIPES AND PIPE FITTINGS, GUTTERS AND GUTTER FITTINGS AND ROOF FITTINGS PART 2 GUTTERS AND GUTTER FITTINGS Note \u2014 For methods",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "AND ROOF FITTINGS PART 2 GUTTERS AND GUTTER FITTINGS (Second Revision) Note \u2014 For methods of tests, refer to IS 5913:1989 Methods of tests for asbestos cement products (first revision). For detailed information, refer to IS 1626 (Part 2) : 1994 Specification for asbestos cement building pipes and pi",
    },
    "IS 1626: 1984": {
        "title":    "ASBESTOS CEMENT BUILDING PIPES AND PIPE FITTINGS, GUTTERS AND GUTER FITTINGS AND ROOF FITTINGS PART 3 ROOF FITTINGS Note \u2014 For methods of tests, refer",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "AND ROOF FITTINGS PART 3 ROOF FITTINGS (Second Revision) Note \u2014 For methods of tests, refer to IS 5913:1989 Methods of tests for asbestos cement products (first revision). For detailed information, refer to IS 1626(Part 3):1994 Specification for asbestos cement building pipes and pipe fittings, gutt",
    },
    "IS 2096: 1992": {
        "title":    "ASBESTOS CEMENT FLAT SHEETS .2 On length and width \u2014 Shall not vary from the nomoinal dimensions for length and width by more than \u00b1 5 mm. .3 Straight",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "3.3.2 On length and width \u2014 Shall not vary from the nomoinal dimensions for length and width by more than \u00b1 5 mm. 3.3.3 Straightness of edges \u2014 Shall be not more than 2 mm/m for the relevant dimension (length or width) 3.3.4 Squareness of edges \u2014 Shall be not more than",
    },
    "IS 2098: 1997": {
        "title":    "ASBESTOS CEMENT BUILDING BOARDS b) From 6 mm and above \u00b1 e mm (\u00b1 10 percent) where 'e' is nominal thickness of board. 4. Tests Load Bearing Capacity\u2014",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "b) From 6 mm and above \u00b1 0.1 e mm (\u00b1 10 percent) where 'e' is nominal thickness of board. 4. Tests 4.1 Load Bearing Capacity\u2014 Average of two specimens not less than 20 kg for Class A boards and",
    },
    "IS 6908: 1991": {
        "title":    "ASBESTOS CEMENT PIPES AND FITTINGS FOR SEWERAGE AND DRAINAGE Fittings \u2014 Tolerances on the nominal thickness of the fittings shall be as follows: Upper",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "(First revision) 5.2 Fittings \u2014 Tolerances on the nominal thickness of the fittings shall be as follows: Upper deviation : Free Lower deviation",
    },
    "IS 8870: 1978": {
        "title":    "ASBESTOS CEMENT CABLE CONDUITS AND TROUGHS TABLE 1 DIMENSIONS AND PERMISSIBLE VARIATIONS OF ASBESTOS CEMENT CONDUITS AND BEND Nominal Length Permissib",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "TABLE 1 DIMENSIONS AND PERMISSIBLE VARIATIONS OF ASBESTOS CEMENT CONDUITS AND BEND Nominal Length Permissible Variation Nominal Internal",
    },
    "IS 9627: 1980": {
        "title":    "ASBESTOS CEMENT PRESSURE PIPES (LIGHT DUTY) 4. Dimensions and Tolerances Nominal diameters and other dimension of pipes\u2014 Shall be given in Table 1. To",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "4. Dimensions and Tolerances 4.1 Nominal diameters and other dimension of pipes\u2014 Shall be given in Table 1. 4.2 Tolerances\u2014 a) Diameter\u2014 \u00b1 0.6 mm b) Thickness\u2014",
    },
    "IS 13000: 1990": {
        "title":    "SILICA ASBESTOS - CEMENT FLAT SHEETS Note\u2014 For methods of tests, refer to Methods of test for asbestos cement products . For detailed information refe",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Note\u2014 For methods of tests, refer to IS 5913:1989 Methods of test for asbestos cement products (first revision). For detailed information refer to IS 13000:1990 Specification for silica-asbestos-cement flat sheets. Length Width mm",
    },
    "IS 13008: 1990": {
        "title":    "SHALLOW CORRUGATED ASBESTOS CEMENT SHEETS Impermeability \u2014 Shall not show during 24 hours of test any formation of drops of water except traces of moi",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "3.2 Impermeability \u2014 Shall not show during 24 hours of test any formation of drops of water except traces of moisture on the lower surface. 4. Finish \u2014 Shall have a rectangular shape, smooth surface on the weathering side, a good apearance and shall be true and regular. The edges of the sheets shall",
    },
    "IS 458: 2003": {
        "title":    "PRECAST CONCRETE PIPES (WITH AND WITHOUT REINFORCEMENT) \u2013 SPECIFICATION 1. Scope \u2014 Requirements for reinforced and unreinforced precast cement concret",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements for reinforced and unreinforced precast cement concrete pipes, of both pressure and non- pressure varieties used for water mains, sewers, culverts and irrigation. The requirements for collars are also covered by this standard.",
    },
    "IS 784: 2001": {
        "title":    "PRESTRESSED CONCRETE PIPES (INCLUDING SPECIALS) 1. Scope \u2014 Requirements of prestressed concrete cylinder and non- cylinder pipes (including specials)",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements of prestressed concrete cylinder and non- cylinder pipes (including specials) with nominal internal diameter in the range of 200 mm to",
    },
    "IS 4350: 1967": {
        "title":    "CONCRETE POROUS PIPES FOR UNDER DRAINAGE .1 Deviation from straightness \u2014 Not to exceed 3 mm per metre run. 3. Tests Load Test \u2014 Specimen shall suppor",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "2.4.1 Deviation from straightness \u2014 Not to exceed 3 mm per metre run. 3. Tests 3.1 Load Test \u2014 Specimen shall support a minimum load of 2000 kg uniformly distributed per metre length of pipe without showing any signs of failure at least for 1",
    },
    "IS 7319: 1974": {
        "title":    "PERFORATED CONCRETE PIPES 3. Sizes and Dimensions \u2014 See Table-1 Tolerances \u2014 Table 2 4. Workmanship and Finish Shall be free from fractures, cracks an",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "3.1 Tolerances \u2014 Table 2 4. Workmanship and Finish 4.1 Shall be free from fractures, cracks and blisters laminations and surface roughness. 4.2 Joints \u2014 Spigot and socket type. 4.3. Specials \u2014 shall have spigot and socket ends.",
    },
    "IS 7322: 1985": {
        "title":    "SPECIALS FOR STEEL CYLINDER REINFORCED CONCRETE PIPES 1. Scope \u2014 Requirements and methods of tests for steel cylinder reinforced concrete specials for",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Requirements and methods of tests for steel cylinder reinforced concrete specials for steel cylinder reinforced concrete pipes conforming to IS 1916 : 1989* having nominal internal diameter from 200 to 1800mm. Covers special having\u2014 a) Spigot and socket ends, b) Plain ends or slip- in type ends suitable for field welding, and c) Flanged ends for connection with valves and accessories.",
    },
    "IS 1834: 1984": {
        "title":    "HOT APPLIED SEALING COMPOUNDS FOR JOINTS IN CONCRETE TABLE 1 PHYSICAL REQUIREMENTS OF SEALING COMPOUNDS OF GRADES A AND B Sl No. Characteristic Requir",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "(First Revision) TABLE 1 PHYSICAL REQUIREMENTS OF SEALING COMPOUNDS OF GRADES A AND B Sl No. Characteristic Requirement (1)",
    },
    "IS 1838 (PART 2): 1984": {
        "title":    "PREFORMED FILLERS FOR EXPANSION JOINT IN CONCRETE PAVEMENT AND STRUCTURES (NON-EXTRUDING AND RESILIENT TYPE) PART 2 CNSL ALDEHYDE RESIN AND COCONUT PI",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "Specifies the materials, manufacture, properties and tests for CNSL aldehyde resin and coconut pith based fillers for expansion joints in concrete roads, runways, bridges and other structures.",
    },
    "IS 11433 (PART 1): 1985": {
        "title":    "ONE-PART GUN-GRADE POLYSULPHIDE- BASED JOINT SEALANTS PART 1 GENERAL REQUIREMENTS 1. Scope\u2014 General requirements of one-part gun- grade polysulphide-b",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "General requirements of one-part gun- grade polysulphide-based sealants used in some sealing or glazing applications in buildings and structures.",
    },
    "IS 12118 (PART 1): 1987": {
        "title":    "TWO-PARTS POLYSULPHIDE BASED SEALANTS PART 1 GENERAL REQUIREMENTS the original profile in a horizontal position. Plastic Deformation\u2014 The sealant shal",
        "category": "Cement and Concrete",
        "section":  1,
        "scope":    "General requirements of two grades of two-part polysulphide based sealants for use in general building applications, namely, pouring grade and gun grade. Pouring grade sealants are intended for use in horizental joints. Gun grade sealants are intended for use in vertical and inclined joints (that is, glazing applications).",
    },
    "IS 712: 1984": {
        "title":    "Building Limes",
        "category": "Building Limes",
        "section":  2,
        "scope":    "Requirements for building limes including fat lime, hydraulic lime and eminently hydraulic lime (Class A, B, C, D, E) used in construction.",
    },
    "IS 2686: 1977": {
        "title":    "CINDER AS FINE AGGREGATES FOR USE IN LIME CONCRETE For detailed information, refer to Specification for cinder as fine aggregates for use in lime conc",
        "category": "Building Limes",
        "section":  2,
        "scope":    "(First Revision) For detailed information, refer to IS 2686 :1977 Specification for cinder as fine aggregates for use in lime concrete(first revision). Note \u2014 For methods of tests, refer to Appendices A and B of the standard.",
    },
    "IS 3068: 1986": {
        "title":    "BROKEN BRICK (BURNT CLAY) COARSE AGGREGATE FOR USE IN LIME CONCRETE Note \u2014 For methods of tests, refer to Appendices A and B of the standard and Metho",
        "category": "Building Limes",
        "section":  2,
        "scope":    "(Second Revision) Note \u2014 For methods of tests, refer to Appendices A and B of the standard and IS 2386: 1963 Methods of tests for aggregates for concrete. IS 5640: 1970 Methods of test for determining aggregate impact value of soft coarse aggregates. For detailed information, refer to IS 3068:1986 S",
    },
    "IS 3115: 1992": {
        "title":    "LIME BASED BLOCKS Note \u2014 For methods of test, refer to Concrete masonry units\u2014 Part 1 solid and hollow concrete blocks . For detailed information, ref",
        "category": "Building Limes",
        "section":  2,
        "scope":    "Note \u2014 For methods of test, refer to IS 2185 (Part 1):1979 Concrete masonry units\u2014 Part 1 solid and hollow concrete blocks (second revision). For detailed information, refer to IS 3115:1992 Specification for lime based blocks (second revision ). 4.2 Tolerances\u2014 Length \u00b1 5 mm, Max",
    },
    "IS 3182: 1986": {
        "title":    "BROKEN BRICK (BURNT CLAY) FINE AGGREGATE FOR USE IN LIME MORTAR 4. Requirement of Broken Brick Fine Aggregate Specific gravity - Clay and silt, percen",
        "category": "Building Limes",
        "section":  2,
        "scope":    "(Second Revision) 4. Requirement of Broken Brick Fine Aggregate Specific gravity 2.4 - 2.7 Clay and silt, percent, Max",
    },
    "IS 4098: 1983": {
        "title":    "LIME POZZOLANA MIXTURE TABLE 2 PHYSICAL REQUIREMENTS. Sl Characteristic Requirment No. Type of Mixture",
        "category": "Building Limes",
        "section":  2,
        "scope":    "TABLE 2 PHYSICAL REQUIREMENTS. Sl Characteristic Requirment No. Type of Mixture",
    },
    "IS 4139: 1989": {
        "title":    "CALCIUM SILICATE BRICKS 1. Scope\u2014Requirements regarding classification, general quality, dimensions, compressive strength and drying shrikage of calci",
        "category": "Building Limes",
        "section":  2,
        "scope":    "Requirements regarding classification, general quality, dimensions, compressive strength and drying shrikage of calcium silicate bricks used in building.",
    },
    "IS 10360: 1982": {
        "title":    "LIME-POZZOLANA CONCRETE BLOCKS FOR PAVING Note \u2014 For method of tests, refer to Concrete masonry units Part 1 Hollow and solid concrete blocks Burnt cl",
        "category": "Building Limes",
        "section":  2,
        "scope":    "(Second Revision) Note \u2014 For method of tests, refer to IS 2185 (Part 1): 1997 Concrete masonry units Part 1 Hollow and solid concrete blocks IS 2690 (Part 2):1992 Burnt clay flat terracing tiles: Part 2 Handmade (second revision), and IS 9284:1979 Method of test for abrasion resistance of concrete,",
    },
    "IS 10772: 1983": {
        "title":    "QUICK SETTING LIME POZZOLANA MIXTURE Note \u2014For methods of tests, refer to various parts of Methods of sampling and tests for quicklime and hydrated li",
        "category": "Building Limes",
        "section":  2,
        "scope":    "(first revision) IS 1727:1967 Method of test for pozzolanic material (first revision). Various parts of IS 4031 Methods of physical tests for hydraulic cement IS 4098:1983 Lime-pozzolana mixture (first revision), and Various parts of IS 6932 Methods of test for building limes. For detailed informati",
    },
    "IS 12894: 2002": {
        "title":    "Pulverised Fuel Ash-Lime Bricks",
        "category": "Building Limes",
        "section":  2,
        "scope":    "Requirements for bricks manufactured from pulverised fuel ash (fly ash) and lime, with or without the addition of other materials, for use in masonry construction.",
    },
    "IS 1127: 1970": {
        "title":    "DIMENSIONS AND WORKMANSHIP OF NATURAL BUILDING STONES FOR MASONRY WORK 1. Scope \u2014 Recommendations for the dimensions and workmanship of natural buildi",
        "category": "Stones",
        "section":  3,
        "scope":    "Recommendations for the dimensions and workmanship of natural building stones used for various types of stone masonry.",
    },
    "IS 1128: 1974": {
        "title":    "LIMESTONE (SLAB AND TILES) TABLE 1 STANDARD SIZES OF LIMESTONE SLABS AND TILES Length Breadth Thickness (1) (2)",
        "category": "Stones",
        "section":  3,
        "scope":    "TABLE 1 STANDARD SIZES OF LIMESTONE SLABS AND TILES Length Breadth Thickness (1) (2)",
    },
    "IS 1130: 1969": {
        "title":    "MARBLE (BLOCKS, SLABS AND TILES) 1. Scope \u2014 Requirements for sizes, physical properties, quality and workmanship of marble (block, slabs and tiles) 2.",
        "category": "Stones",
        "section":  3,
        "scope":    "Requirements for sizes, physical properties, quality and workmanship of marble (block, slabs and tiles)",
    },
    "IS 3316: 1974": {
        "title":    "STRUCTURAL GRANITE",
        "category": "Stones",
        "section":  3,
        "scope":    "STRUCTURAL GRANITE 1. Scope \u2014 Requirements for dimensions, physical properties and workmanship of rectangular blocks made from laterite stone, used in the construction of walls and partitions. 2. General Requirements \u2014 Shall be exposed for three months before using but not to rains. Shall be without any soft veins, cracks, cavities, flaws and similar imperfections. 3. Dimensions \u2014 Length Breadth T",
    },
    "IS 3620: 1979": {
        "title":    "LATERITE STONE BLOCK FOR MASONRY Note \u2014 For methods of tests, refer to Methods of test for determination of strength properties of natural building st",
        "category": "Stones",
        "section":  3,
        "scope":    "Note \u2014 For methods of tests, refer to IS 1121(Part 1) : 1974 Methods of test for determination of strength properties of natural building stones : Part 1 Compressive strength (first revision) and IS 1124:1974 Method of test for determination of water absorption, apparent specific gravity and porosit",
    },
    "IS 3622: 1977": {
        "title":    "SANDSTONE (SLABS AND TILES) Note \u2014 The sizes in between (of length and breadth) shall be reckoned as next lower size. This aspect will also cover tole",
        "category": "Stones",
        "section":  3,
        "scope":    "Note \u2014 The sizes in between (of length and breadth) shall be reckoned as next lower size. This aspect will also cover tolerance in length and breadth. 3.1.1 Tolerances \u2014 The tolerance for thickness shall be \u00b1 3 mm. 3.2 Machine Cut Slabs \u2014 Machine cut slabs with true",
    },
    "IS 6250: 1981": {
        "title":    "ROOFING SLATE TILES 1. Scope \u2014 Requirements of dimensions, physical properties and workmanship of slate tiles used for sloped roof covering. Requireme",
        "category": "Stones",
        "section":  3,
        "scope":    "Requirements of dimensions, physical properties and workmanship of slate tiles used for sloped roof covering. Requirements in regard to method of laying and fixing of slate tiles for roofing covered in IS 5119 (Part 1):1968*.",
    },
    "IS 6579: 1981": {
        "title":    "COARSE AGGREGATE FOR WATER BOUND MACADAM Note \u2014 For methods of tests refer to relevant parts of Methods of test for aggregates for concrete and Method",
        "category": "Stones",
        "section":  3,
        "scope":    "(First Revision) Note \u2014 For methods of tests refer to relevant parts of IS 2386:1963 Methods of test for aggregates for concrete and IS 5640:1970 Method of test for determining aggregate impact value of soft course aggregates. For detailed information, refer to IS 6579:1981. Specification for coarse",
    },
    "IS 9394: 1979": {
        "title":    "STONE LINTELS *Recommendation for dressing of natural building stones .",
        "category": "Stones",
        "section":  3,
        "scope":    "*Recommendation for dressing of natural building stones",
    },
    "IS 14223 (PART 1): 1995": {
        "title":    "POLISHED BUILDING STONES PART 1 GRANITE Note \u2014 For methods of test, refer to relevant parts of Methods of test for determination of strength propertie",
        "category": "Stones",
        "section":  3,
        "scope":    "Note \u2014 For methods of test, refer to relevant parts of IS 1121:1974 Methods of test for determination of strength properties of natural building stones, (first revision) IS 1124:1974 Methods of test for dertermination of water absorption, apparent specific gravity and proposity of natural building s",
    },
    "IS 1077: 1992": {
        "title":    "Common Burnt Clay Building Bricks",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "Requirements for common burnt clay building bricks used in construction of masonry. Covers dimensions, physical requirements and classification into classes.",
    },
    "IS 2180: 1988": {
        "title":    "HEAVY DUTY BURNT CLAY BUILDING BRICKS 4. Dimensions 190 mm \u00d7 90 mm \u00d7 90 mm, and 190 mm \u00d7 90 mm \u00d7 40 mm 5. Tolerances",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "4. Dimensions 190 mm \u00d7 90 mm \u00d7 90 mm, and 190 mm \u00d7 90 mm \u00d7 40 mm 5. Tolerances\u2014",
    },
    "IS 2222: 1991": {
        "title":    "Burnt Clay Perforated Building Bricks",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "Requirements for burnt clay perforated building bricks used in construction of walls and partitions. Covers dimensions, compressive strength and water absorption.",
    },
    "IS 2691: 1988": {
        "title":    "BURNT CLAY FACING BRICKS 5. Physical Requirements Average Compressive Strength shall not be less than 10N/mm2 Water absorption after 24 hours immersio",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "5. Physical Requirements 5.1 Average Compressive Strength shall not be less than 10N/mm2 5.2 Water absorption after 24 hours immersion shall not exceed 15 percent.",
    },
    "IS 3583: 1988": {
        "title":    "BURNT CLAY PAVING BRICKS",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "BURNT CLAY PAVING BRICKS 1. Scope \u2014 Covers the dimensions, quality and strength requirements of hollow bricks made from burnt clay and having perforations through and at right angle to the bearing surface. 2. General Requirements Bricks shall be free from cracks, flaws and nodules of free lime. Shall be of uniform colour. Shall have plane rectangular faces with parallel sides and shall have sharp",
    },
    "IS 3952: 1988": {
        "title":    "Burnt Clay Hollow Bricks for Walls and Partitions",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "Requirements for burnt clay hollow bricks used in construction of load-bearing and non-load bearing walls and partitions. Covers dimensions and physical properties.",
    },
    "IS 4885: 1988": {
        "title":    "SEWER BRICKS Note \u2014 For method of the tests refer to the relevant parts of Method of test for burnt clay building bricks . For detailed information, r",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "Specifies dimensions, quality and strength, and methods of sampling and test for burnt clay sewer bricks used for sewers of sanitary (domestic) sewage.",
    },
    "IS 5779: 1986": {
        "title":    "BURNT CLAY SOLING BRICKS",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "BURNT CLAY SOLING BRICKS 1. Scope \u2014 Dimensions for special shapes of clay brick used in building and other civil engineering construction. It does not lay down the specification of the special shapes for clay bricks and same shall conform to * and IS: 2180:1988\u2020. Note \u2014For exact shape of clay bricks and detailed dimensions, refer to Fig. 1 to 7 of the standard. For detailed information, refer to S",
    },
    "IS 6165: 1992": {
        "title":    "DIMENSIONS FOR SPECIAL SHAPES OF CLAY BRICKS Shape Major Overall Dimensions m m a) Closers",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "(First Revision) Shape Major Overall Dimensions m m a) Closers \u2014",
    },
    "IS 13757: 1993": {
        "title":    "BURNT CLAY FLY ASH BUILDING BRICKS * Heavy duty burnt clay building bricks . ** Common burnt day building bricks .",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "** Common burnt day building bricks (fifth revision).",
    },
    "IS 7556: 1988": {
        "title":    "BURNT CLAY JALLIES",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "BURNT CLAY JALLIES 1. Scope \u2014 Covers the machine-pressed clay interlocking roofing tiles of the \u2018Mangalore Pattern.\u2019 2. Classification \u2014 Class AA and Class A with characteristics given in Table 1. TABLE 1 CLASSIFICATION OF ROOFING TILES Sl.no. Characteristic Requirement Class AA Class A i) Water absorption percent, Max 18 20 ii) Breaking load, kN, Min a)Average (for 410 \u00d7 235 mm) (for 410 \u00d7 235 mm",
    },
    "IS 654: 1992": {
        "title":    "CLAY ROOFING TILES, MANGALORE PATTERN . For measurement of variations in length/width of tiles the difference between\u2014 a) The overall length/width of",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "4.1. For measurement of variations in length/width of tiles the difference between\u2014 a) The overall length/width of three tiles and b) The length/width of a tile is calculated",
    },
    "IS 1464: 1992": {
        "title":    "CLAY RIDGE AND CEILING TILES",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "CLAY RIDGE AND CEILING TILES 3. General Quality \u2014 shall be free from irregularities, such as twists, bends, cracks, flaws, laminations and imperfections. Faces of tiles shall be plain, grooved fluted or figured as specified and the edges shall be square. 4. Dimensions i) 150 \u00d7 150 \u00d7 15 mm ii) 150 \u00d7 150 \u00d7 20 mm iii) 200 \u00d7 200 \u00d7 20 mm iv) 200 \u00d7 200 \u00d7 25 mm v) 250 \u00d7 250 \u00d7 30 mm Depth of the grooves o",
    },
    "IS 1478: 1992": {
        "title":    "CLAY FLOORING TILES",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "CLAY FLOORING TILES 1. Scope \u2014 Requirements for machine-made burnt clay flat terracing tiles. 2. General Quality \u2014 Shall be uniform in shape and sizes and shall be free from irregularities, such as twists, bends, cracks and particles of stones. 3. Dimensions and Toleranes Length \u2014 250 to 150 mm in stages of 25 mm. Width \u2014 200 to 100 mm in stages of 25 mm. Thickness \u201420 and 15 mm. Tolerances \u2014 \u00b1 2",
    },
    "IS 2690 (PART 1): 1993": {
        "title":    "BURNT CLAY\u2013FLAT TERRACING TILES PART 1 MACHINE-MADE",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "BURNT CLAY\u2013FLAT TERRACING TILES PART 1 MACHINE-MADE",
    },
    "IS 2690 (PART 2): 1992": {
        "title":    "BURNT CLAY FLAT TERRACING TILES PART 2 HAND \u2013 MADE Note \u2014 For methods of test, refer to Annex B of the standard, and relevant parts of IS 3495 Methods",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "(Second Revision) Note \u2014 For methods of test, refer to Annex B of the standard, and relevant parts of IS 3495 Methods of tests of burnt clay building bricks (third revision). For detailed information, refer to IS 2690 (Part 2):1992 Specification for burnt clay flat terracing tiles: Part 2 Hand-made",
    },
    "IS 3367: 1993": {
        "title":    "BURNT CLAY TILES FOR USE IN LINING IRRIGATION AND DRAINAGE WORKS",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "BURNT CLAY TILES FOR USE IN LINING IRRIGATION AND DRAINAGE WORKS",
    },
    "IS 3951 (PART 1): 1975": {
        "title":    "HOLLOW CLAY TILES FOR FLOORS AND ROOFS PART 1 FILLER TYPE",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "PART 1 FILLER TYPE (First Revision)",
    },
    "IS 3951 (PART 2): 1975": {
        "title":    "HOLLOW CLAY TILES FOR FLOORS AND ROOFS PART 2 STRUCTURAL TYPE",
        "category": "Clay Products for Building",
        "section":  4,
        "scope":    "PART 2 STRUCTURAL TYPE (First Revision)",
    },
    "IS 2095 (PART 1): 1996": {
        "title":    "GYPSUM PLASTER BOARDS PART 1 PLAIN GYPSUM PLASTER BOARDS 4. General\u2014 Gypsum plaster boards consist of a gypsum plaster core with or without fibre enca",
        "category": "Gypsum Building Materials",
        "section":  5,
        "scope":    "(Second Revision) 4. General\u2014 Gypsum plaster boards consist of a gypsum plaster core with or without fibre encased in and firmly bonded to strong durable paper liners to form rectangular boards. Core shall be dried across full width. The face and back papers shall be securely bonded to",
    },
    "IS 2095 (PART 3): 1996": {
        "title":    "GYPSUM PLASTER BOARDS PART 3 REINFORCED GYPSUM PLASTER BOARDS 1. Scope \u2014 Covers the method of manufacture, tests and sampling of fibrous gypsum plaste",
        "category": "Gypsum Building Materials",
        "section":  5,
        "scope":    "Covers the method of manufacture, tests and sampling of fibrous gypsum plaster boards and glass fibre reinforced gypsum (GRG) boards for use as a linning material for ceiling, dry surfacing material for walls, door panels or for partitions.",
    },
    "IS 2547 (PART 1): 1976": {
        "title":    "GYPSUM BUILDING PLASTERS PART 1 EXCLUDING PREMIXED LIGHTWEIGHT PLASTERS",
        "category": "Gypsum Building Materials",
        "section":  5,
        "scope":    "PART 1 EXCLUDING PREMIXED LIGHTWEIGHT PLASTERS",
    },
    "IS 2547 (PART 2): 1976": {
        "title":    "GYPSUM BUILDING PLASTER PART 2 PREMIXED LIGHTWEIGHT PLASTERS Note\u2014For methods of tests, refer to Appendices A and B of the standard and relevant parts",
        "category": "Gypsum Building Materials",
        "section":  5,
        "scope":    "(First Revision) Note\u2014For methods of tests, refer to Appendices A and B of the standard and relevant parts of IS 2542 Methods of test for Gypsum plaster, Concrete and Products. For detailed information, refer to IS 2547 (Part 2): 1976 Specification for gypsum building plasters: Part 2 Premixed light",
    },
    "IS 2849: 1983": {
        "title":    "NON-LOAD BEARING GYPSUM PARTITION BLOCKS (SOLID AND HOLLOW TYPES) 1. Scope \u2014 Requirements for gypsum partition blocks for use in non-load bearing cons",
        "category": "Gypsum Building Materials",
        "section":  5,
        "scope":    "Requirements for gypsum partition blocks for use in non-load bearing construction in the interior of buildings and for the protection of columns, elevator shafts, etc, against fire.",
    },
    "IS 8272: 1984": {
        "title":    "GYPSUM PLASTER FOR USE IN THE MANUFACTURE OF FIBROUS PLASTER BOARDS 3. Properties Fineness \u2014 Residue retained on 600 micron sieve shall not be more th",
        "category": "Gypsum Building Materials",
        "section":  5,
        "scope":    "(First Revision) 3. Properties 3.1 Fineness \u2014 Residue retained on 600 micron sieve shall not be more than 1 percent by mass. 3.2 Compressive Strength \u2014 Compressive strength of the plaster, shall not be less than",
    },
    "IS 399: 1963": {
        "title":    "CLASSIFICATION OF COMMERCIAL TIMBERS AND THEIR ZONAL DISTRIBUTION (Revised) 4. Classification \u2014 Tables I, II, III, IV, and V of the standard list resp",
        "category": "Timber",
        "section":  6,
        "scope":    "(Revised) 4. Classification \u2014 Tables I, II, III, IV, and V of the standard list respectively important timbers commercially available in the five zones described under 3 and classified according to their uses given under 2. Against each species of timber, the availability in that zone,",
    },
    "IS 12896: 1990": {
        "title":    "INDIAN TIMBERS FOR DOOR AND WINDOW SHUTTERS AND FRAMES \u2013 CLASSIFICATION 1. Scope \u2013 Covers the general classification of Indian timber species suitable",
        "category": "Timber",
        "section":  6,
        "scope":    "Covers the general classification of Indian timber species suitable for door and window shutters and frames. It also lays down the general requirements of quality, seasoning, moisture content and preservative treatmesnt for timber. This standard does not, however, cover the species suitable for flush doors.",
    },
    "IS 190: 1991": {
        "title":    "CONIFEROUS SAWN TIMBER (BAULKS AND SCANTLINGS) metres correct to three places of decimals. 5. Requirements\u2014 Shall be air seasoned to a moisture conten",
        "category": "Timber",
        "section":  6,
        "scope":    "(Fourth Revision) metres correct to three places of decimals. 5. Requirements\u2014 Shall be air seasoned to a moisture content not exceeding 20 percent within a depth of 15 mm from the surface, excluding a l e n g t h of 300 mm from each end.",
    },
    "IS 876: 1992": {
        "title":    "WOOD POLES FOR OVER HEAD POWER AND TELECOMMUNICATION LINES Class 5 - Ultimate breaking load not less than 5 500 N and not more than 7 000 N. Class 6 -",
        "category": "Timber",
        "section":  6,
        "scope":    "(Third Revision) Class 5 - Ultimate breaking load not less than 5 500 N and not more than 7 000 N. Class 6 - Ultimate breaking load not less than 4 000 N",
    },
    "IS 1326: 1992": {
        "title":    "NON-CONIFEROUS SAWN TIMBER (BAULKS AND SCANTLING) of 13 mm from the surface, excluding 300 mm from each end. Timber shall be either sawn or axe-hewn.",
        "category": "Timber",
        "section":  6,
        "scope":    "(Second Revision) of 13 mm from the surface, excluding 300 mm from each end. Timber shall be either sawn or axe-hewn. Any axe-hewn timber shall be reasonably even. All pieces shall have fairly straight and parallel sides and rectangular cross",
    },
    "IS 2372: 2004": {
        "title":    "TIMBER FOR COOLING TOWERS For detailed information, refer to Specification for timber for cooling towers .",
        "category": "Timber",
        "section":  6,
        "scope":    "For detailed information, refer to IS 2372:2004 Specification for timber for cooling towers (Second revision).",
    },
    "IS 3337: 1978": {
        "title":    "BALLIES FOR GENERAL PURPOSES For detailed information, refer to Specifications for ballies for general purposes. End Cracks Spiral or Twisted Grain Cu",
        "category": "Timber",
        "section":  6,
        "scope":    "For detailed information, refer to IS 3337: 1978 Specifications for ballies for general purposes. 6.2 End Cracks 6.3 Spiral or Twisted Grain 6.4 Curvature 6.5 Short Crooks 6.6 Pin Hole (Dead Infestation)\u2014For extent of defects",
    },
    "IS 3629: 1986": {
        "title":    "STRUCTURAL TIMBER IN BUILDINGS 1. Scope \u2014 Covers the various requirements of structural timber for use in buildings. It includes classification and gr",
        "category": "Timber",
        "section":  6,
        "scope":    "Covers the various requirements of structural timber for use in buildings. It includes classification and grouping of different species of timber, their suitability for permanent and temporary structures, factors affecting strength, tolerances on dimensions, influence of defects and allowance for such defects in timber.",
    },
    "IS 3731: 1985": {
        "title":    "TEAK SQUARES 1. Scope \u2014 Covers the requirements of various grades of teak squares based on defects. 2. Grades Grade 1 \u2014 No single square shall contain",
        "category": "Timber",
        "section":  6,
        "scope":    "Covers the requirements of various grades of teak squares based on defects.",
    },
    "IS 4895: 1985": {
        "title":    "TEAK LOGS 1. Scope \u2014 Covers the requiremens of various grades of teak logs intended for conversion purposes. It does not cover the requirements of tea",
        "category": "Timber",
        "section":  6,
        "scope":    "Covers the requiremens of various grades of teak logs intended for conversion purposes. It does not cover the requirements of teak logs for veneering purposes.",
    },
    "IS 5246: 2000": {
        "title":    "CONIFEROUS LOGS For detailed Information, refer to Specification for coniferous logs first revision",
        "category": "Timber",
        "section":  6,
        "scope":    "For detailed Information, refer to IS 5246 : 2000 Specification for coniferous logs first revision",
    },
    "IS 6056: 1970": {
        "title":    "JOINTED WOOD POLES FOR OVERHEAD POWER TELECOMMUNICATION LINES TABLE 1 DIMENSION OF THE JOINTED WOOD POLES Overall Groundline Minimum Circumference at",
        "category": "Timber",
        "section":  6,
        "scope":    "TABLE 1 DIMENSION OF THE JOINTED WOOD POLES Overall Groundline Minimum Circumference at Ground LinePosition Height Position",
    },
    "IS 7308: 1999": {
        "title":    "NON-CONIFEROUS LOGS For detailed information, refer to Specification for non-coniferous logs .",
        "category": "Timber",
        "section":  6,
        "scope":    "For detailed information, refer to IS 7308 : 1999 Specification for non-coniferous logs (first revision).",
    },
    "IS 10394: 1982": {
        "title":    "WOODEN SLEEPERS FOR RAILWAY TRACK For detailed information, refer to Specification for wooden sleeper for railway track TABLE 1 DIMENSIONS FOR STANDAR",
        "category": "Timber",
        "section":  6,
        "scope":    "track TABLE 1 DIMENSIONS FOR STANDARD TRACK SLEEPERS Gauge Length Tolerance Cross Sectional",
    },
    "IS 73: 1992": {
        "title":    "PAVING BITUMEN Penetrationat C g s Penetrationat C",
        "category": "Bitumen and Tar Products",
        "section":  7,
        "scope":    "Penetrationat C g s Penetrationat C",
    },
    "IS 212: 1983": {
        "title":    "CRUDE COAL TAR FOR GENERAL USE TABLE 1 REQUIREMENTS OF CRUDE COAL TAR Sl. No. Characteristics Min Max (1) (2)",
        "category": "Bitumen and Tar Products",
        "section":  7,
        "scope":    "TABLE 1 REQUIREMENTS OF CRUDE COAL TAR Sl. No. Characteristics Min Max (1) (2)",
    },
    "IS 215: 1995": {
        "title":    "ROAD TAR roads at high altitudes as well as for priming the base; RT-2 \u2014 For surface painting in normal climatic conditions; RT-3 \u2014 a) For surface pai",
        "category": "Bitumen and Tar Products",
        "section":  7,
        "scope":    "roads at high altitudes as well as for priming the base; RT-2 \u2014 For surface painting in normal climatic conditions; RT-3 \u2014 a) For surface painting and renewal coat;",
    },
    "IS 216: 1961": {
        "title":    "COAL TAR PITCH (Revised) TABLE 1 EQUIREMENTS FOR COAL TAR PITCH Sl. Characteristics Requirements for Grades No. Soft Pitch",
        "category": "Bitumen and Tar Products",
        "section":  7,
        "scope":    "TABLE 1 EQUIREMENTS FOR COAL TAR PITCH Sl. Characteristics Requirements for Grades No. Soft Pitch",
    },
    "IS 218: 1983": {
        "title":    "CREOSOTE OIL FOR USE AS WOOD PRESERVATIVES 1. Scope\u2013 Covers materials commercially known as coal tar creosote (or creosote oil) primarily used for pre",
        "category": "Bitumen and Tar Products",
        "section":  7,
        "scope":    "Covers materials commercially known as coal tar creosote (or creosote oil) primarily used for preservation of wood.",
    },
    "IS 454: 1994": {
        "title":    "CUTBACK BITUMEN FROM WAXY CRUDE 1. Scope\u2014 Covers the physical and chemical requirements of cutbacks bitumen from waxy crude of indigenous origin. 2. G",
        "category": "Bitumen and Tar Products",
        "section":  7,
        "scope":    "Covers the physical and chemical requirements of cutbacks bitumen from waxy crude of indigenous origin.",
    },
    "IS 3117: 2004": {
        "title":    "BITUMEN EMULSION FOR ROADS AND ALLIED APPLICATIONS (ANIONIC TYPE) (First Revision ) 1. Scope \u2014 Physical and chemical requirements of grades of bitumen",
        "category": "Bitumen and Tar Products",
        "section":  7,
        "scope":    "Physical and chemical requirements of grades of bitumen emulsion (anionic type) for roads and allied applcations.",
    },
    "IS 8887: 2004": {
        "title":    "BITUMEN EMULSION FOR ROADS (CATIONIC TYPE) 1. Scope \u2014 Covers the physical and chemical requirements of bitumen emulsions (cationic type) for roads. 2.",
        "category": "Bitumen and Tar Products",
        "section":  7,
        "scope":    "Covers the physical and chemical requirements of bitumen emulsions (cationic type) for roads.",
    },
    "IS 1237: 1980": {
        "title":    "Cement Concrete Flooring Tiles",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Requirements for cement concrete floor tiles used for flooring in buildings. Covers dimensions, finish, transverse strength and water absorption.",
    },
    "IS 1542: 1992": {
        "title":    "Sand for Plaster",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Requirements for sand used in plastering work for buildings. Covers naturally occurring sands, crushed stone sands and crushed gravel sands.",
    },
    "IS 4457: 1982": {
        "title":    "CERAMIC UNGLAZED VITREOUS ACID RESISTING TILES Note \u2014 For methods of tests, refer to Appendices of the standard. For detailed information, refer to Sp",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Requirements for ceramic unglazed vitreous acid resisting titles.",
    },
    "IS 4832 (PART 1): 1969": {
        "title":    "CHEMICAL RESISTANT MORTARS PART I \u2013 SILICATE TYPE Note 1\u2014 For method of tests, refer to Methods of test for chemical resistant mortar: Part I Silicate",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Note 1\u2014 For method of tests, refer to IS 4456 (Part 1) : 1967 Methods of test for chemical resistant mortar: Part I Silicate type and resin type Note 2\u2014 For general guide for chemical resistance of sillicate type mortars to various substances, refer to Table 1 of IS 4441:1980 Code of practice for us",
    },
    "IS 4832 (PART 2): 1969": {
        "title":    "CHEMICAL RESISTANT MORTARS PART 2 RESIN TYPE TABLE 1 PHYSICAL REQUIREMENTS OF RESIN TYPE CHEMICAL RESISTANT MORTARS Sl No. Particular Requirements for",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "TABLE 1 PHYSICAL REQUIREMENTS OF RESIN TYPE CHEMICAL RESISTANT MORTARS Sl No. Particular Requirements for Type of Mortar Phenolic",
    },
    "IS 4832 (PART 3): 1968": {
        "title":    "CHEMICAL RESISTANT MORTARS PART 3 \u2013 SULPHUR TYPE TABLE 1 PHYSICAL REQUIRE- MENTS OF SULPHUR TYPE CHEMI- CAL RESISTANT MORTARS S.No. Property Requireme",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "TABLE 1 PHYSICAL REQUIRE- MENTS OF SULPHUR TYPE CHEMI- CAL RESISTANT MORTARS S.No. Property Requirement",
    },
    "IS 4860: 1968": {
        "title":    "ACID \u2013 RESISTANT BRICKS 3. Performance Requirements\u2013See Table 1 4. Dimensions \u2014 230 \u00d7 114 \u00d7 64 mm. 5. Tolerances Dimensions Tolerances (mm) (mm)",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "4. Dimensions \u2014 230 \u00d7 114 \u00d7 64 mm. 5. Tolerances Dimensions Tolerances (mm) (mm)",
    },
    "IS 13753: 1993": {
        "title":    "DUST\u2013 PRESSED CERAMIC TILES WITH WATER ABSORPTION OF E>10% GROUP B III 1. Scope \u2014 Specifies sizes, dimensional tolerances, mechanical, physical and ch",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Specifies sizes, dimensional tolerances, mechanical, physical and chemical requirements, surface quality requirements and marking of ceramic tiles. 1.1It is applicable only to dust-pressed ceramic glazed tiles first quality, with a water absorption (E>10%) according to Group B III of IS 13712 : 1993* for use as both wall and floor coverings. Tiles in this group are mainly used in areas not subject to severe mechanical load. They are not intended ",
    },
    "IS 13754: 1993": {
        "title":    "DUST \u2013 PRESSED CERAMIC TILES WITH WATER ABSORPTION OF 6% < E \u2264 10% (GROUP B II B ) Characteristics A) Dimensions and Surface Quality i) Length and wid",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "(GROUP B II B ) Characteristics A) Dimensions and Surface Quality i) Length and width\u2014 The deviation in % of the average size of each tile (2 to 4 sides) from the work size.",
    },
    "IS 13755: 1993": {
        "title":    "DUST\u2013 PRESSED CERAMIC TILES WITH WATER ABSORPTION OF 3% < E \u2264 6% (GROUP \u2013 B II A) *Ceramictiles\u2014 definitions,classification, characteristics and marki",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "(GROUP \u2013 B II A) *Ceramictiles\u2014 definitions,classification, characteristics and marking TABLE 1 REQUIREMENTS Characteristics Surface S of the Product (cm2) A) Dimensions and Surface Quality S< 90 - 90<S< 190 1 90 < S < 410 S>410",
    },
    "IS 13756: 1993": {
        "title":    "DUST \u2013 PRESSED CERAMIC TILES WITH LOW WATER ABSORPTION OF E 3% GROUP B1 1. Scope \u2014Specifies the sizes,dimensional tolerances, mechanical, physical and",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Specifies the sizes,dimensional tolerances, mechanical, physical and chemical requirements, surface quality requirements and marking of ceramic tiles. It is applicable only to dust-pressed ceramic tiles including tiles premounted on sheets of first quality, with a low water absorption (E \u2264 3%) according to Group BI of IS 13712 : 1993 Ceramic tiles\u2013 Efinitions, Classifications, Characteristics and marking. For interior and exterior use on both flo",
    },
    "IS 14862: 2000": {
        "title":    "FIBRE CEMENT FLAT SHEETS 1. Scope This standard covers the characteristics and establishes methods of control and test as well as acceptance condition",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "1.1 This standard covers the characteristics and establishes methods of control and test as well as acceptance conditions for fibre cement flat sheets. It covers sheets intended for external applications, such as cladding facades, curtain walls, soffits, etc, and sheets intended for internal use, such as partitions, floors, ceiling, etc, with a wide range of properties appropriate to the type of application. These sheets may have either a smooth ",
    },
    "IS 14871: 2000": {
        "title":    "PRODUCTS IN FIBRE REINFORCED CEMENT\u2014LONG CORRUGATED OR ASYMMETRICAL SECTION SHEETS AND FITTINGS FOR ROOFING AND CLADDING TABLE 1 CATEGORY AND CLASS (M",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "SHEETS AND FITTINGS FOR ROOFING AND CLADDING TABLE 1 CATEGORY AND CLASS (MINIMUM BREAKING LOAD IN N/M) Category Minimum Class Thickness,",
    },
    "IS 1195: 2002": {
        "title":    "BITUMEN MASTIC FOR FLOORING .1 Fine Aggregate \u2014 The fine aggregate shall con- sist of naturally occuring limestone rock ground to a grading as given i",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "3.2.1 Fine Aggregate \u2014 The fine aggregate shall con- sist of naturally occuring limestone rock ground to a grading as given in Table 2, and shall have a calcium carbonate content of not less than 80 percent by weight 3.2.2. Coarse Aggregate : The coarse aggregates shall consist of clean igneous or c",
    },
    "IS 5317: 2002": {
        "title":    "BITUMEN MASTIC FOR BRIDGE DECKING AND ROADS 1. Scope \u2014 Requirements for bitumen mastic used as a surfacing material for bridge decks and roads. 2. Mat",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Requirements for bitumen mastic used as a surfacing material for bridge decks and roads.",
    },
    "IS 8374: 1977": {
        "title":    "BITUMEN MASTIC, ANTI-STATIC AND ELECTRICALLY CONDUCTING GRADE 1. Scope \u2014 Requirements of bitumen mastic for anti-static and electrically conducting gr",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Requirements of bitumen mastic for anti-static and electrically conducting grade.",
    },
    "IS 13026: 1991": {
        "title":    "BITUMEN MASTIC FOR FLOORING FOR INDUSTRIES HANDLING LPG AND OTHER LIGHT HYDROCARBON PRODUCTS",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "INDUSTRIES HANDLING LPG AND OTHER LIGHT",
    },
    "IS 653: 1992": {
        "title":    "LINOLEUM SHEET AND TILES",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "LINOLEUM SHEET AND TILES TABLE 1 REQUIREMENT OF NOLEUM SHEETS, LINOLEUM TILES AND CORK TILES Sl.No Characteristic Requirement (1) ( 2) (3) i) Width of sheet Average value shall not vary by more than \u00b13 mm ii) Tolerance to tile size \u00b1 percent iii) Thickness Average value shall not vary by more than + m iv) Squareness Gap between thesides (for tiles only) of tileand arms of the metal jig ,shall not",
    },
    "IS 3461: 1980": {
        "title":    "PVC ASBESTOS FLOOR TILES iv) Volatile matter Loss in weight shall not exceed 1 percent. v) Curling Shall not exceed mm vi) Indentation",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "iv) Volatile matter Loss in weight shall not exceed 1 percent. v) Curling Shall not exceed 0.75 mm vi) Indentation",
    },
    "IS 3462: 1986": {
        "title":    "UNBACKED FLEXIBLE PVC FLOORING 1. Scope \u2014 Requirements of unbacked homogeneous flexible PVC flooring, including laminated",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Requirements of unbacked homogeneous flexible PVC flooring, including laminated PVC flooring in which the composition of each of the laminate is substantially the same. The flooring may be supplied in continuous lengths or in tile form.",
    },
    "IS 9197: 1979": {
        "title":    "EPOXY RESIN, HARDENERS AND EPOXY RESIN COMPOSITIONS FOR FLOOR TOPPING } Accelerator \u2014 Liquids generally tertiary amines. Plasticizers and Non-reactive",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "} 2.3 Accelerator \u2014 Liquids generally tertiary amines. 2.4 Plasticizers and Non-reactive Diluents \u2014 May be incorporated in the resins and hardeners, pro- vided the total quantity of these ingredients does not exceed 25 parts per hundred parts by weight of the resin,",
    },
    "IS 12866: 1989": {
        "title":    "PLASTIC TRANSLUCENT SHEET MADE FROM THERMOSETTING POLYESTER RESIN (GLASS FIBRE REINFORCED) TABLE 1 DIMENSION AND TOLERANCES OF GLASSFIBRE REINFORCED C",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "(GLASS FIBRE REINFORCED) TABLE 1 DIMENSION AND TOLERANCES OF GLASSFIBRE REINFORCED CORRUGATED TRANSLUCENT ROOFLIGHT SHEETS All dimensions in millimetres. Sl. Type of",
    },
    "IS 638: 1979": {
        "title":    "SHEET RUBBER JOINTING AND RUBBER INSERTION JOINTING 1. Scope \u2014 Requirements and the methods of sampling and test for sheet rubber jointing and rubber",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "Requirements and the methods of sampling and test for sheet rubber jointing and rubber insertion jointing for use between flanges and similar joints subjected to water pressure, air pressure or low pressure steam.",
    },
    "IS 809: 1992": {
        "title":    "RUBBER FLOORING MATERIALS FOR GENERAL PURPOSE * Indian Hessian, Part 2-305 and 229g/m2 at 16 percent contact regain",
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "section":  8,
        "scope":    "(First Revision) * Indian Hessian, Part 2-305 and 229g/m2 at 16 percent contact regain (first revision)",
    },
    "IS 1322: 1993": {
        "title":    "BITUMEN FELTS FOR WATER\u2013PROOFING AND DAMP\u2013PROOFING 1. Scope\u2014Requirements for saturated bitumen felts (underlay) and self-finished bitumen felts used f",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "Requirements for saturated bitumen felts (underlay) and self-finished bitumen felts used for water proofing and damp-proofing.",
    },
    "IS 1580: 1991": {
        "title":    "BITUMINOUS COMPOUNDS FOR WATER-PROOFING AND CAULKING PURPOSES 1. Scope \u2014 Requirements and methods of sampling and tests for bituminous compound, appli",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "Requirements and methods of sampling and tests for bituminous compound, applied cold and used for stopping leaks through cracks of roofs, floors, walls, etc; as sealant for plate joints of wagons, coaches and buses; as caulking agent for crevices and vertical joints between steel plates, folded sections, wood joints, precast concrete cladding, etc; and as adhesives for rainguards for rubber trees.",
    },
    "IS 2645: 2003": {
        "title":    "INTEGRAL CEMENT WATER-PROOFING COMPOUNDS 1. Scope \u2014 Requirements for integral cement water- proofing compounds, which shall be assessed by: a) Permeab",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "Requirements for integral cement water- proofing compounds, which shall be assessed by: a) Permeability to water, and b) Physical tests of setting time and compressive strengths of cement mixed with the water-proofing compounds.",
    },
    "IS 3037: 1986": {
        "title":    "BITUMEN MASTIC FOR USE IN WATER\u2014PROOFING OF ROOFS 1. Scope \u2014 Requirements for bitumen mastic suitable for water proofing of roofs. This bitumen mastic",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "Requirements for bitumen mastic suitable for water proofing of roofs. This bitumen mastic is not intended to be used as a paving material or to with stand exceptional conditions, such as acid or alkali actions.",
    },
    "IS 3384: 1986": {
        "title":    "BITUMEN PRIMER FOR USE IN WATER\u2013PROOFING AND DAMP\u2013PROOFING TABLE 1 REQUIREMENTS OF PRIMER Sl.No. Characteristic Requirement (1)",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "(First Revision) TABLE 1 REQUIREMENTS OF PRIMER Sl.No. Characteristic Requirement (1)",
    },
    "IS 5871: 1987": {
        "title":    "BITUMEN MASTIC FOR TANKING AND DAMP\u2013PROOFING 1. Scope \u2014 Requirements for bitumen mastic used as covering material for damp-proofing of underground tan",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "Requirements for bitumen mastic used as covering material for damp-proofing of underground tanks, basements of building, water reservoirs, swimming pools, irrigation canals, etc.",
    },
    "IS 7193: 1974": {
        "title":    "GLASS FIBRE BASE BITUMEN FELTS 1. Scope \u2014 Requirements for self finished glass fibre bitumen felts used for waterproofing and damp proofing. Note \u2014 Gl",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "Requirements for self finished glass fibre bitumen felts used for waterproofing and damp proofing.",
    },
    "IS 12027: 1987": {
        "title":    "SILICONE-BASED WATER REPELLENTS 3. Consistency \u2014 The water repellent shall be of such consistency that it can be readily applicable to masonary by bru",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "such consistency that it can be readily applicable to masonary by brushing or spraying. 4. Performance Requirement 4.1 Early Water Repellency \u2014 Water repellency shall be such that no pool of water shall be completely absorbed within 10 minutes.",
    },
    "IS 14695: 1999": {
        "title":    "GLASS FIBRE BASE COAL TAR PITCH OUTER WRAP Note\u2014 For test procedure, refer to Methods of test for bitumen based felts: (Part 1) Breaking strength test",
        "category": "Waterproofing and Damp Proofing Materials",
        "section":  9,
        "scope":    "Note\u2014 For test procedure, refer to IS 13826 : 1993 Methods of test for bitumen based felts: (Part 1) Breaking strength test. (Part 2) Pliability test. For detailed information, refer to IS 14695 : 1999 Specification for glass fibre base coal tar pitch outer wrap. 3. Other Requirements \u2014 See Table 2",
    },
    "IS 775: 1970": {
        "title":    "CAST IRON BRACKETS AND SUPPORTS FOR WASH-BASINS AND SINKS",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "CAST IRON BRACKETS AND SUPPORTS FOR WASH-BASINS AND SINKS",
    },
    "IS 782: 1978": {
        "title":    "CAULKING LEAD 1. Scope \u2014 Requirements for different types of caulking lead suitable for use in water supply and sanitary installations. 2. Type a) Pig",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for different types of caulking lead suitable for use in water supply and sanitary installations.",
    },
    "IS 804: 1967": {
        "title":    "RECTANGULAR PRESSED STEEL TANKS 1. Scope \u2014 Requirements for the materials, fabrication, erection and supply for rectangular pressed steel tanks used f",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for the materials, fabrication, erection and supply for rectangular pressed steel tanks used for the storage of cold and hot water and certain other liquids under pressure not greater than the static head corresponding to the depth of the tank. This specification does not cover the requirements of tanks storing liquids having temperature higher than 100\u00b0C, or those tanks subject to earth or other external pressure besides wind pressu",
    },
    "IS 1700: 1973": {
        "title":    "DRINKING FOUNTAINS 1. Scope \u2014 Covers the material, construction, essential hygienic and performance requirements and finish of drinking fountains used",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Covers the material, construction, essential hygienic and performance requirements and finish of drinking fountains used in schools, parks and other public places.",
    },
    "IS 2963: 1979": {
        "title":    "COPPER ALLOY WASTE FITTINGS FOR WASH-BASINS AND SINKS 1. Scope\u2014 Requirements for materials, manufacture and workmanship, nominal sizes, dimensions and",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for materials, manufacture and workmanship, nominal sizes, dimensions and finish of copper alloy waste fittings used in wash-basins and sinks complying with the prescribed standards.",
    },
    "IS 3489: 1985": {
        "title":    "ENAMELLED STEEL BATH TUBS 1. Scope\u2014Requirements for material, construction and workmanship, patterns, dimensions, tolerances and maintenance for vitre",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for material, construction and workmanship, patterns, dimensions, tolerances and maintenance for vitreous enamelled steel bath tubs.",
    },
    "IS 5219: 1969": {
        "title":    "CAST COPPER ALLOYS TRAPS PART 1 P AND S TRAPS 1. Scope \u2013 Covers copper alloy cast traps P and S types and their associated components of nominal sizes",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Covers copper alloy cast traps P and S types and their associated components of nominal sizes 32 mm, 40 mm and 50 mm for use in wash\u2013basins, sinks, bath tubs and similar waste appliances.",
    },
    "IS 8718: 1978": {
        "title":    "VITREOUS ENAMELLED STEELKITCHEN SINKS 1. Scope : Requirmements regading material construction and workmanship, patterns and sizes, dimensions and tole",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    ": Requirmements regading material construction and workmanship, patterns and sizes, dimensions and tolerance and marking for vitreous enamelled steel kitchen sinks. Sl. Pattem Overall Overall OverallDepth (in mm) No. length Width mm mm Min Max mm mm i) Flat-rim 750 450 150 200 600 450 150 200 500 400 150 200 450 400 150 200 400 400 150 200 ii) Flat-rim-ledge 750 500 150 200 600 500 150 200 iii) Flat-rim- ledge, 1050 500 150 200 with doule compart",
    },
    "IS 8727: 1978": {
        "title":    "VITREOUS ENAMELLED STEEL WASH\u2013BASINS Note \u2014 For test procedures refer to General requirements for enamelled cast iron sanitary appliances, and IS 3972",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "revision), and IS 3972. Methods of test for vitreous enamel ware. For detailed information, refer to IS 8727 : 1978 Specification for vitreous enamelled steel wash-basins.",
    },
    "IS 12701: 1996": {
        "title":    "ROTATIONAL MOULDED POLYETHYLENE WATER STORAGE TANKS asdfa FIG. 1 TYPICAL DETAILS OF CYLINDRICAL VERTICAL TANK",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) asdfa FIG. 1 TYPICAL DETAILS OF CYLINDRICAL VERTICAL TANK",
    },
    "IS 13983: 1994": {
        "title":    "STAINLESS STEELSINKS FOR DOMESTIC PURPOSES",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "STAINLESS STEELSINKS FOR DOMESTIC PURPOSES",
    },
    "IS 407: 1981": {
        "title":    "BRASS TUBES FORGENERAL PURPOSES",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "BRASS TUBES FORGENERAL PURPOSES",
    },
    "IS 2501: 1995": {
        "title":    "SOLID DRAWN COPPER TUBES FOR GENERAL ENGINEERING PURPOSES",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "FOR GENERAL ENGINEERING PURPOSES",
    },
    "IS 1230: 1979": {
        "title":    "CAST IRON RAINWATER PIPES AND FITTINGS Note 1 \u2014 For dimens of bends, shoes, branches, offsets, union sockets, holderbats, rainwater heads, refer to th",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Note 1 \u2014 For dimens of bends, shoes, branches, offsets, union sockets, holderbats, rainwater heads, refer to the standard. Note 2 \u2014 For test details, refer to the standard and IS 1500:1983 Methods for brinell hardness test for mettallic materials (second revision). For detailed information, refer to",
    },
    "IS 1536: 2001": {
        "title":    "CENTRIFUGALLY CAST (SPUN) IRON PRESSURE PIPES FOR WATER, GAS AND SEWAGE",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PIPES FOR WATER, GAS AND SEWAGE",
    },
    "IS 1537: 1976": {
        "title":    "VERTICALLY CAST IRON PRESSURE PIPES FOR WATER, GAS AND SEWAGE \u00b1",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "VERTICALLY CAST IRON PRESSURE PIPES FOR WATER, GAS AND SEWAGE \u00b1",
    },
    "IS 1538: 1993": {
        "title":    "CAST IRON FITTINGS FOR PRESSURE PIPES FOR WATER, GAS AND SEWAGE 1. Scope General requirements for cast iron fittings for pressure pipes for water, gas",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "1.1 General requirements for cast iron fittings for pressure pipes for water, gas and sewage. 1.2 Applicable to all cast iron fittings having spigots, sockets or flanges as specified in this standard and also to fittings with other type of joints, the general dimensions of which, except those relating to the joints, conform to this standard.",
    },
    "IS 2002: 1979": {
        "title":    "SAND CAST IRON SPIGOT AND SOCKET SOIL, WASTE AND VENTILATING PIPES, FITTINGS AND ACCESSORIES Short Radius Bends with and without Access Doors Nominal",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Second Revision) 3.2 Short Radius Bends with and without Access Doors Nominal Size \u03b8 (Bend) 50, 75,100,150, 92\u00bd\u00b0, 95\u00b0, 100\u00b0, 112\u00bd\u00b0 ,120\u00b0, 135\u00b0,",
    },
    "IS 1879: 1987": {
        "title":    "MALLEABLE CAST IRON PIPE FITTINGS + Malleable Iron Castings",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "MALLEABLE CAST IRON PIPE FITTINGS + Malleable Iron Castings",
    },
    "IS 3989: 1984": {
        "title":    "CENTRIFUGALLY CAST (SPUN) IRON SPIGOT AND SOCKET SOIL WASTE AND VENTILATING PIPES FITTINGS AND ACCESSORIES { \u03b8",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "SOCKET SOIL WASTE AND VENTILATING PIPES FITTINGS AND ACCESSORIES",
    },
    "IS 5382: 1985": {
        "title":    "RUBBER SEALIG RINGS FOR GAS MAINS, WATER MAINS AND SEWERS Finish \u2014 The rings shall be homogeneous; free from porosity, grit, excessive blooms, blister",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) 3.2 Finish \u2014 The rings shall be homogeneous; free from porosity, grit, excessive blooms, blisters or other visible surface imperfections. The fin or flash shall be reduced as much as possible and in any case the thickness of it shall be reduced as much as possible and",
    },
    "IS 5531: 1988": {
        "title":    "CAST IRON SPECIALS FOR ASBESTOS CEMENT PRESSURE PIPES FOR WATER, GAS AND SEWAGE 1. Scope \u2014 Requirements for cast iron specials to be used with asbesto",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for cast iron specials to be used with asbestos cement pressure pipes for water, gas and sewage. 1.2 Applicable to cast iron specials for use with asbestos cement pressure pipes suitable for connection with cast iron detachable joints or asbestos cement couplings.",
    },
    "IS 6163: 1978": {
        "title":    "CENTRIFUGALLY CAST (SPUN) IRON LOW PRESSURE PIPES FOR WATER, GAS AND SEWAGE 5. Sizes (in mm) Sockets and Spigots of Low Pressure Pipes (Lead Joint) No",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for centrifugally cast (spun) iron low pressure pipes, known as LP pipes, for conveyance of water, gas and sewage, manufactured in metal or sand moulds. 1.2 This standard is applicable to cast iron pipes having spigots and sockets as specified in this standard, and also to pipes with other types of joints particularly rubber joints. In case of rubber joints the inner profile of the socket end of the pipe shall depend on the type of r",
    },
    "IS 6418: 1971": {
        "title":    "CAST IRON AND MALLEABLE CAST IRON FLANGES FOR GENERAL ENGINEERING PURPOSES . Type of gasket and gasket materials are not covered in the standard and s",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "1.2. Type of gasket and gasket materials are not covered in the standard and shall be subject to agreement between the manufacturer and the purchaser. 2. Pressure and Temperature Rating \u2014 Table1 Nominal Typeof Material",
    },
    "IS 7181: 1986": {
        "title":    "HORIZONTALLY CAST IRON DOUBLE FLANGED PIPES FOR WATER, GAS AND SEWAGE \u00b1 \u00b1 \u00b1",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PIPES FOR WATER, GAS AND SEWAGE",
    },
    "IS 8329: 2000": {
        "title":    "CENTRIFUGALLY CAST (SPUN) DUCTILE IRON PRESSURE PIPES FOR WATER, GAS AND SEWAGE . Fittings conforming to * may also be used with ductile iron pipes, w",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) 1.5. Fittings conforming to IS 13382 : 1992* may also be used with ductile iron pipes, when the pressure requirements matches. 2. Classification 2.1 K7, K8, K9, K10, K12, ..... depending on service",
    },
    "IS 8794: 1988": {
        "title":    "CAST IRON DETACHABLE JOINTS FOR USE WITH ASBESTOS CEMENT PRESSURE PIPES Nominal Class External Dia Dia of",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) Nominal Class External Dia Dia of",
    },
    "IS 9523: 2000": {
        "title":    "DUCTILE IRON FITTINGS FOR PRESSURE PIPES FOR WATER, GAS AND SEWAGE *Iron castings with spheroidal modular or modular graphite For detailed information",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) *Iron castings with spheroidal modular or modular graphite (third revision) For detailed information, refer to IS 9523 : 1980 Specification for ductile iron fittings, for pressure pipes for water gas and sewage.",
    },
    "IS 10292: 1988": {
        "title":    "DIMENSIONAL REQUIREMENTS FOR RUBBER SEALING RINGS FOR C I D. JOINTS IN ASBESTOS CEMENT PIPING TABLE 1 DIMENSIONS OF RUBBER SEALING RINGS Nominal Dia o",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "ASBESTOS CEMENT PIPING (First Revision) TABLE 1 DIMENSIONS OF RUBBER SEALING RINGS Nominal Dia of Class Inner Dia",
    },
    "IS 10299: 1982": {
        "title":    "CAST IRON SADDLE PIECES FOR SERVICE CONNECTION FROM ASBESTOS CEMENT PRESSURE PIPES 3. Tests . Tensile test\u2014 Minimum 150 MPa. Brinell Hardness \u2014 Not mo",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "3. Tests 3.1. Tensile test\u2014 Minimum 150 MPa. 3.2 Brinell Hardness \u2014 Not more than 215 HB. 4. Dimensions see Table 1 TABLE 1 DIMENSIONS FOR SADDLE PIECES Nominal",
    },
    "IS 12820: 2004": {
        "title":    "DIMENSIONAL REQUIREMENTS OF RUBBER GASKETS FOR MECHANICAL JOINTS AND PUSH-ON JOINTS FOR USE WITH CAST IRON PIPES AND FITTINGS FOR CARRYING WATER, GAS",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "FOR USE WITH CAST IRON PIPES AND FITTINGS FOR CARRYING WATER, GAS AND SEWAGE (First Revision) TABLE 2 DIMENSIONS OF RUBBER GASKETS FOR MECHANICAL JOINT",
    },
    "IS 12987: 1991": {
        "title":    "CAST IRON DETACHABLE JOINTS FOR USE WITH ASBESTOS CEMENT PRESSURE PIPES (LIGHT DUTY) Note \u2014 Nominal diameter of detachable joint shall refer to the co",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Note \u2014 Nominal diameter of detachable joint shall refer to the corresponding nominal diameter of the asbestos cement pressure pipes. 5. Coatings 5.1 Coating shall not be applied to any part unless its surface is clean, dry and free from rust",
    },
    "IS 12988: 1991": {
        "title":    "DIMENSIONAL REQUIREMENTS FOR RUBBER SEALING RINGS FOR CID JOINTS FOR LIGHT DUTY AC PIPES \u2013 DIMENSIONAL REQUIREMENTS For detailed information, refer to",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "AC PIPES \u2013 DIMENSIONAL REQUIREMENTS For detailed information, refer to IS 12988 : 1991 Specification for rubber sealing rings for CID joints for light duty AC pipes\u2013Dimensional requirements. * Rubber sealing rings for gas mains, water mains and sewers (first revision). \u2020 Asbestos cement pressure pip",
    },
    "IS 13382: 2004": {
        "title":    "CAST IRON SPECIALS FOR MECHANICAL AND PUSH ON FLEXIBLE JOINTS FOR PRESSURE PIPE LINES FOR WATER, GAS AND SEWAGE The bolt hole circles shall be concent",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "WATER, GAS AND SEWAGE (First revision) 3.3 The bolt hole circles shall be concentric with the bore and shall be located off the centre line, unless otherwise specified by the purchaser. Where there are two or more flanges, the bolt holes shall be correctly",
    },
    "IS 11925: 1986": {
        "title":    "PITCH IMPREGNATED FIBRE PIPES AND FITTINGS FOR DRAINGAGE PURPOSES 1. Scope \u2014 Covers materials, dimension and methods of testing of pitch impregnated f",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Covers materials, dimension and methods of testing of pitch impregnated fibre pipes and fittings in diametrs ranging from 50 to 200 mm for drainage purposes below and above ground level. 1.2 The standard also covers perforated pipes of the same materials for sub-surface drainage.",
    },
    "IS 404 (PART 1): 1993": {
        "title":    "LEAD PIPES \u2013 FOR OTHER THAN CHEMICAL PURPOSES 1. Scope \u2014 Requirements of lead pipes for other than chemical purposes. The lead pipes covered in this s",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements of lead pipes for other than chemical purposes. The lead pipes covered in this standard are not suitable for potable water supply.",
    },
    "IS 3076: 1985": {
        "title":    "LOW DENSITY POLYETHYLENE PIPES FOR POTABLE WATER SUPPLIES 1. Scope \u2014 Requirements for low density black polyethylene pipes of outside diameters up to",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for low density black polyethylene pipes of outside diameters up to 140 mm for use in potable water supplies.",
    },
    "IS 4984: 1995": {
        "title":    "HIGH DENSITY POLYETHYLENE PIPES FOR WATER SUPPLY * High density polyethylene malerials for molding and extension . \u00b1",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Fourth Revision) * High density polyethylene malerials for molding and extension (first revision). \u00b1",
    },
    "IS 4985: 2000": {
        "title":    "UNPLASTICIZED PVC PIPES FOR POTABLE WATER SUPPLIES DIMENSION OF UPVC PIPES Nominal outside Mean outside Diameter mm Diameter mm",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Third Revision) DIMENSION OF UPVC PIPES Nominal outside Mean outside Diameter mm Diameter mm",
    },
    "IS 7834: 1987": {
        "title":    "INJECTION MOULDED PVC SOCKET FITTINGS WITH SOLVENT CEMENT JOINTS FOR WATER SUPPLIES PART \u2013 1 GENERAL REQUIREMENT 1. Scope \u2014 General requirements regar",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "General requirements regarding materials, manufacture, methods of test, inspection and marking of all types of injection moulded PVC socket fittings intended for connection, by using solvent cement, to PVC pipes covered by IS 4985 : 1988 [Specification for unplasticized PVC pipes for potable water supplies (second revision) for water supplies.",
    },
    "IS 7834 (PART 2): 1987": {
        "title":    "INJECTION MOULDED PVC SOCKET FITTINGS WITH SOLVENT CEMENT JOINTS FOR WATER SUPPLIES PART 2 SPECIFIC REQUIREMENTS FOR 45\u00b0 ELBOWS. \u2013 1 63 14 + \u2013 1",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "SUPPLIES PART 2 SPECIFIC REQUIREMENTS FOR 45\u00b0 ELBOWS. (First Revision) \u2013 1 63 14 + 3.2 \u2013 1",
    },
    "IS 7834 (PART 3): 1987": {
        "title":    "INJECTION MOULDED PVC SOCKET FITTINGS WITH SOLVENT CEMENT JOINTS FOR WATER SUPPLIES PART-3 SPECIFIC REQUIREMENTS FOR 900 ELBOWS First Revision Note\u2014 F",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "WATER SUPPLIES PART-3 SPECIFIC REQUIREMENTS FOR 900 ELBOWS First Revision Note\u2014 For typical illustration of 90\u00b0 elbow see Fig.1 of the standard. For detailed information, refer to IS :7834 (Part 3) : 1987 Specification for injection moulded PVC socket fittings with solvent cement joints for water su",
    },
    "IS 7834 (PART 4): 1987": {
        "title":    "INJECTION MOULDED PVC SOCKET FITTINGS WITH SOLVENT CEMENT JOINTS FOR WATER SUPPLIES PART 4 SPECIFIC REQUIREMENTS FOR 90\u00b0 TEES 1. Scope \u2013 Requirements",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for manufacture, dimensions, tolerances and marking for 900 tee made of injection moulded PVC for water supplies.",
    },
    "IS 7834 (PART 5): 1987": {
        "title":    "INJECTION MOULDED PVC SOCKET FITTINGS WITH SOLVENT CEMENT JOINTS FOR WATER SUPPLIES PART \u2013 5 SPECIFIC REQUIREMENTS FOR 450 TEES. 1. Scope \u2014Requirement",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for manufacture, dimensions, tolerances and marking for 450 tees made of injections moulded PVC for water supplies.",
    },
    "IS 7834 (PART 6): 1987": {
        "title":    "INJECTION MOULDING PVC SOCKET FITTINGS WITH SOLVENT CEMENT JOINTS FOR WATER SUPPLIES PART 6 SPECIFIC REQUIREMENTS FOR SOCKETS Note \u2013 For typical illus",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 6 SPECIFIC REQUIREMENTS FOR SOCKETS (First Revision) Note \u2013 For typical illustration of socket and Z see fig. 1 of the standard. For detailed information, refer to IS 7834 (Part 6) : 1987 Specification for Injection moulded PVC socket fittings with solvent cement joints for water supplies : Par",
    },
    "IS 7834 (PART 7): 1987": {
        "title":    "INJECTION MOULDED PVC SOCKET FITINGS WITH SOLVENT CEMENT JOINTS FOR WATER SUPPLIES PART 7 SPECIFIC REQUIREMENTS FOR UNIONS. The inside diameter of the",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 7 SPECIFIC REQUIREMENTS FOR UNIONS. (First Revision) 2.2 The inside diameter of the socket and the length shall comply with those given in IS 7834 (Part 1):1987*",
    },
    "IS 7834 (PART 8): 1987": {
        "title":    "INJECTION MOULDED PVC SOCKET FITTINGS WITH SOLVENT CEMENT JOINTS FOR WATER SUPPLIES PART 8 SPECIFIC REQUIREMENTS FOR CAPS 1. Scope \u2014 Requirements for",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for manufacture, dimensions, tolerances and marking for caps made of injections moulded PVC for water supplies.",
    },
    "IS 8008 (PART 1): 2003": {
        "title":    "INJECTION MOULDED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 1 GENERAL REQUIREMENTS FOR FITTINGS Note \u2013 For test proced",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 1 GENERAL REQUIREMENTS FOR FITTINGS (First Revision) Note \u2013 For test procedures refer to annex. B and C of IS 4984:1995. For detailed information, refer to IS: 8008(Part I ):2003. Specification for injection moulded high density polyethylene (HDPE) fittings for potable water supplies : Part I G",
    },
    "IS 8008 (PART 2): 2003": {
        "title":    "INJECTION MOULDED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 2 SPECIFIC REQUIREMENTS FOR 90O BENDS",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 2 SPECIFIC REQUIREMENTS FOR 90O BENDS (First Revision)",
    },
    "IS 8008 (PART 3): 2003": {
        "title":    "INJECTION MOULDED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 3 SPECIFIC REQUIREMENTS FOR 90O TEES 40 63 \u00b1 2 50 75 \u00b1 2",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 3 SPECIFIC REQUIREMENTS FOR 90O TEES (First Revision) 40 63 \u00b1 2 50 75 \u00b1 2",
    },
    "IS 8008 (PART 4): 2003": {
        "title":    "INJECTION MOULDED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 4 SPECIFIC REQUIREMENTS FOR REDUCERS Size Bends Laying Len",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 4 SPECIFIC REQUIREMENTS FOR REDUCERS (First Revision) Size Bends Laying Length mm mm",
    },
    "IS 8008 (PART 5): 2003": {
        "title":    "INJECTION MOULDED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 5 SPECIFIC REQUIREMENTS FOR FERRULE REDUCERS",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 5 SPECIFIC REQUIREMENTS FOR FERRULE REDUCERS (First Revision)",
    },
    "IS 8008 (PART 6): 2003": {
        "title":    "INJECTION MOULDED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 6 SPECIFIC REQUIREMENTS FOR PIPE ENDS Note \u2014For general re",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 6 SPECIFIC REQUIREMENTS FOR PIPE ENDS (First Revision) Note \u2014For general requirements regarding material, manufacture, methods of test, etc, refer to IS 8008 (Part 1) : 2003. Injection moulded high density polyethylene (HDPE) fittings for portable water supplies : Part 1 General requirements fo",
    },
    "IS 8008 (PART 7): 2003": {
        "title":    "INJECTION MOULDED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 7 SPECIFIC REQUIREMENTS FOR SANDWITCH FLANGES TABLE 1 DIME",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 7 SPECIFIC REQUIREMENTS FOR SANDWITCH FLANGES (First Revision) TABLE 1 DIMENSIONS OF SANDWICH FLANGES Sl. Nominal Pipe",
    },
    "IS 8360 (PART 1): 1977": {
        "title":    "FABRICATED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 1 GENERAL REQUIREMENTS outside diameters of pipes given in . Outs",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "WATER SUPPLIES PART 1 GENERAL REQUIREMENTS outside diameters of pipes given in IS 4984 : 1995. Outside diameters and corresponding wall thickness of fittings at free ends for weld shall comply with those given in Table 1 of IS 4984 : 1995",
    },
    "IS 8360 (PART 2): 1977": {
        "title":    "FABRICATED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 2 SPECIFIC REQUIREMENTS FOR 90O TEES Ouside diameters and wall th",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "SUPPLIES PART 2 SPECIFIC REQUIREMENTS FOR 90O TEES 2.2 Ouside diameters and wall thickness of pipes out of which 90\u00b0 tees are fabricated shall comply with those givenin IS 8360 (Part 1) :1977. Wall thickness of a fabricated 90\u00b0 tee shall not be less than that of the pipe to which it i to be welded.",
    },
    "IS 8360 (PART 3): 1977": {
        "title":    "FABRICATED HIGH DENSITY POLYETHYLENE (HDPE) FITTINGS FOR POTABLE WATER SUPPLIES PART 3 SPECIFIC REQUIREMENTS FOR 90\u00b0 BENDS Note\u2014For general requiremen",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 3 SPECIFIC REQUIREMENTS FOR 90\u00b0 BENDS Note\u2014For general requirements regarding material, sizes, methods of test and sampling refer to IS : 8360 (Part 1) :1977. Fabricated high density polyethylene (HDPE) fittings for potable water supplies : Part 1 General Requirements. For detailed information,",
    },
    "IS 10124 (PART 1): 1988": {
        "title":    "FABRICATED PVC FITTINGS FOR POTABLE WATER SUPPLIES PART 1 GENERAL REQUIREMENTS + Unplasticized PVC pipes for potable water supplies Short term Hydraul",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 1 GENERAL REQUIREMENTS (First Revision) + Unplasticized PVC pipes for potable water supplies (third revision) 5.2 Short term Hydraulic Test \u2014 The fittings shall withstand a pressure of 4.2",
    },
    "IS 10124 (PART 2): 1988": {
        "title":    "FABRICATED PVC FITTINGS FOR POTABLE WATER SUPPLIES PART 2 SPECIFIC REQUIREMENTS FOR SOCKETS Note\u2014This figure is only intended to define the terms used",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 2 SPECIFIC REQUIREMENTS FOR SOCKETS (First Revision) Note\u2014This figure is only intended to define the terms used in Table1and is not intended to illustrate specific design features. FIG. 1 SOCKET",
    },
    "IS 10124 (PART 3): 1988": {
        "title":    "FABRICATED PVC FITTINGS FOR PORTABLE WATER SUPPLIES PART 3 SPECIFIC REQUIREMENTS FOR STRAIGHT REDUCER 3. Dimensions \u2014 See Table 1 4. Marking \u2014The stra",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 3 SPECIFIC REQUIREMENTS FOR STRAIGHT REDUCER (First Revision) 3. Dimensions \u2014 See Table 1 4. Marking \u2014The straight reducers shall be marked in colour as indicated below for different classes of fittings",
    },
    "IS 10124 (PART 4): 1988": {
        "title":    "FABRICATED PVC FITTINGS FOR POTABLE WATER SUPPLIES PART 4 SPECIFIC REQUIREMENTS FOR CAPS TABLE 1 DIMENSIONS FOR CAPS ALL DIMENSIONS IN MILLIMETRES Siz",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements of manufacture, dimensions and marking for fabricated PVC caps for potable water supplies.",
    },
    "IS 10124 (PART 5): 1988": {
        "title":    "FABRICATED PVC FITTINGS FOR PORTABLE WATER SUPPLIES PART 5 SPECIFIC REQUIREMENT FOR EQUAL TEES 1. Scope \u2014 Requirements of manufacture, dimensions and",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements of manufacture, dimensions and marking for fabricated PVC tees for potable water supplies.",
    },
    "IS 10124 (PART 6): 1988": {
        "title":    "FABRICATED PVC FITTINGS FOR POTABLE WATER SUPPLIES PART 6 SPECIFIC REQUIREMENTS FOR FLANGED TAIL PIECES WITH METTALLIC FLANGES Note\u2014 This figures is i",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 6 SPECIFIC REQUIREMENTS FOR FLANGED TAIL PIECES WITH METTALLIC FLANGES (First Revision) Note\u2014 This figures is intended to define the terms used in Table 1 and is not intended to illustrate specific design features FIG 1 PVC FLANGED TAIL PIECE WITH METALLIC FLANGE",
    },
    "IS 10124 (PART 7): 1988": {
        "title":    "FABRICATED PVC FITTINGS FOR POTABLE WATER SUPPLIES PART 7 SPECIFIC REQUIREMENT FOR THREADED ADAPTERS FIG. 1 THREADED ADAPTORS",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 7 SPECIFIC REQUIREMENT FOR THREADED ADAPTERS (First Revision) FIG. 1 THREADED ADAPTORS",
    },
    "IS 10124 (PART 8): 1988": {
        "title":    "FABRICATED PVC FITTING FOR POTABLE WATER SUPPLIES PART 8 SPECIFIC REQUIREMENTS FOR 90O BENDS. 4. Marking \u2014The bend shall be marked in colour as indica",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 8 SPECIFIC REQUIREMENTS FOR 90O BENDS. (First Revision) 4. Marking \u2014The bend shall be marked in colour as indicated below for different class of fittings\u2014 Class of Fitting Colour",
    },
    "IS 10124 (PART 9): 1988": {
        "title":    "FABRICATED PVC FITTING FOR POTABLE WATER SUPPLIES PART 9 SPECIFIC REQUIREMENTS FOR 60O BENDS. 4. Marking \u2014The bend shall be marked in colour as indica",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 9 SPECIFIC REQUIREMENTS FOR 60O BENDS. (First Revision) 4. Marking \u2014The bend shall be marked in colour as indicated below for different class of fittings: Class of Fitting Colour",
    },
    "IS 10124 (PART 10): 1988": {
        "title":    "FABRICATED PVC FITTING FOR POTABLE WATER SUPPLIES. PART 10 SPECIFIC REQUIREMENTS FOR 45O BENDS. 4. Marking \u2014The bend shall be marked in colour as indi",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 10 SPECIFIC REQUIREMENTS FOR 45O BENDS. (First Revision) 4. Marking \u2014The bend shall be marked in colour as indicated below for different class of fittings: Class of Fitting",
    },
    "IS 10124 (PART 11): 1988": {
        "title":    "FABRICATED PVC FITTING FOR POTABLE WATER SUPPLIES. PART 11 SPECIFIC REQUIREMENTS FOR 30O BENDS. 4. Marking \u2014The bend shall be marked in colour as indi",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 11 SPECIFIC REQUIREMENTS FOR 30O BENDS. (First Revision) 4. Marking \u2014The bend shall be marked in colour as indicated below for different class of fittings: Class of Fitting Colour",
    },
    "IS 10124 (PART 12): 1988": {
        "title":    "FABRICATED PVC FITTING FOR POTABLE WATER SUPPLIES PART 12 SPECIFIC REQUIREMENTS FOR 22\u00bdO BENDS TABLE 1. DIMENSIONS FOR 22\u00bdO BENDS All Dimensions in mi",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 12 SPECIFIC REQUIREMENTS FOR 22\u00bdO BENDS (First Revision) TABLE 1. DIMENSIONS FOR 22\u00bdO BENDS All Dimensions in millimetres Size Y",
    },
    "IS 10124 (PART 13): 1988": {
        "title":    "FABRICATED PVC FITTING FOR POTABLE WATER SUPPLIES PART 13 SPECIFIC REQUIREMENTS FOR 11\u00bcO BENDS 4. Marking \u2014The bend shall be marked in colour as indic",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 13 SPECIFIC REQUIREMENTS FOR 11\u00bcO BENDS (First Revision) 4. Marking \u2014The bend shall be marked in colour as indicated below for different class of fittings: Class of Fitting Colour",
    },
    "IS 12818: 1992": {
        "title":    "UNPLASTICISED PVC SCREEN AND CASING PIPES FOR BORE/TUBE WELL TABLE 1 DIMENSIONS OF SCREEN PIPES WITH RIBS Nominal Mean Outer OuterDiameter Outer",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) TABLE 1 DIMENSIONS OF SCREEN PIPES WITH RIBS Nominal Mean Outer OuterDiameter Outer",
    },
    "IS 13592: 1992": {
        "title":    "UNPLASTILIZED POLYVINYL CHLORIDE (UPVC) PIPES FOR SOIL AND WASTE DISCHARGE SYSTEM FOR INSIDE AND OUTSIDE BUILDINGS INCLUDING VENTILATION AND RAIN WATE",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "AND OUTSIDE BUILDINGS INCLUDING VENTILATION AND RAIN WATER SYSTEM TABLE 1 DIAMETER AND WALL THICKNESS All dimensions in millimetres Nominal Mean",
    },
    "IS 14333: 1996": {
        "title":    "HIGH DENSITY POLYETHYELENE PIPES FOR SEWERAGE of *, HDPE conforming to designation PEEWA -45-T-012 PEEWA -50-T-012 or PEEWA - 57 - T - 012 of may also",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "of IS 7328 : 1992*, HDPE conforming to designation PEEWA -45-T-012 PEEWA -50-T-012 or PEEWA - 57 - T - 012 of IS 7328 : 1992 may also be used with the exception that melt flow rate (MFR ) shall be between 0.20 g/10 min to 1.10 g / 10 min (both in clusive) 4.1.1 Base density between 940 kg/m 3 and 95",
    },
    "IS 14402: 1996": {
        "title":    "GLASS FIBRE REINFORCED PLASTICS (GRP) PIPES JOINT AND FITTINGS FOR USE FOR SEWERAGE, INDUSTRIAL WASTE AND WATER (OTHER THAN POTABLE) Note\u2014 For details",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "INDUSTRIAL WASTE AND WATER (OTHER THAN POTABLE) Note\u2014 For details of other materials, see 6.3 of the standard. 5. Dimensions 5.1 Inside Diameters and Tolerances \u2014See Table1 TABLE 1 SPECIFIED INSIDE DIAMETERS AND TOLERANCES",
    },
    "IS 14735: 1999": {
        "title":    "UNPLASTICIZED POLYVINYL CHLORIDE (UPVC) INJECTION MOULDED FITTING FOR SOIL AND WASTE DISCHARGE SYSTEM FOR INSIDE AND OUTSIDE BUILDINGS INCLUDING VENTI",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "DISCHARGE SYSTEM FOR INSIDE AND OUTSIDE BUILDINGS INCLUDING VENTILATION AND RAIN WATER SYSTEMS * UPVC pipes for soil and waste discharge systems inside buildings including ventilation and rain water system.",
    },
    "IS 1239 (PART 1): 2004": {
        "title":    "Steel Tubes, Tubulars and Other Wrought Steel Fittings \u2014 Part 1 Steel Tubes",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for welded and seamless wrought steel tubes in the diameter range 6 mm to 150 mm NB for water, gas and steam services and structural purposes.",
    },
    "IS 3589: 2001": {
        "title":    "SEAMLESS OR ELECTRICALLY WELDED STEEL PIPES FOR WATER, GAS AND SEWAGE ( TO 2540 mm OUTSIDE DIAMETER) Note\u2014In case of non-avaibility of ladle analysis,",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(168.3 TO 2540 mm OUTSIDE DIAMETER) (Third Revision) Note\u2014In case of non-avaibility of ladle analysis, the finished product may be checked to verify the chemical composition, if so agreed to by the producer. 5.2 Product Analysis\u2013The permissible variation from",
    },
    "IS 4270: 2001": {
        "title":    "STEEL TUBES USED FOR WATER WELLS",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "STEEL TUBES USED FOR WATER WELLS",
    },
    "IS 5504: 1997": {
        "title":    "SPIRAL WELDED PIPES Flattering Test \u2014 Shall withstand the prescribed test. Submerged Arc Weld Test \u2014 Shall withstand the prescribed test. 5. Hydrostat",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "4.2 Flattering Test \u2014 Shall withstand the prescribed test. 4.3 Submerged Arc Weld Test \u2014 Shall withstand the prescribed test. 5. Hydrostatic Test \u2014 Shall be tested at mill to a hydrostatic pressure, equal to a minimum of 150 percent",
    },
    "IS 6286: 1971": {
        "title":    "SEAMLESS AND WELDED STEEL PIPE FOR SUB-ZERO TEMPERATURE SERVICE 1. Scope \u2013 Requirements for 4 grades of seamless and electric welded steel pipe for co",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for 4 grades of seamless and electric welded steel pipe for conveying fluids at sub-zero temperature.",
    },
    "IS 651: 1992": {
        "title":    "SALT GLAZED STONEWARE PIPES AND FITTINGS 1. Scope \u2013 Covers dimensions and performance requirements for the following glazed stoneware pipes and fittin",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Covers dimensions and performance requirements for the following glazed stoneware pipes and fittings\u2014 Straight pipes and taper pipes; Bends; Taper bend; Junctions; Half-section channels, straight and taper; Channel junctions; Channel bends; Channel interceptors; Gully traps; and Inspection pipes. The pipes covered in this standard are not meant for potable water applications. Dimensions are grouped into two sections A and B. Section A covers dime",
    },
    "IS 3006: 1979": {
        "title":    "CHEMICALLY RESISTANT GLAZED STONEWARE PIPES AND FITTINGS 1. Scope\u2014Material and performance equirements for chemically resistant glazed stoneware pipes",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Material and performance equirements for chemically resistant glazed stoneware pipes (straight pipes) and fittings (taper pipes; bends, taper bends; junctions; half section channels; straight and taper ; channel junctions; channel bends; channel interceptors; gully traps and inspection pipes). Dimensions of chemically resistant glazed stoneware pipes and fittings are grouped into two sections, A and B. Section A covers dimensions of straight pipe",
    },
    "IS 771 (PART 1): 1979": {
        "title":    "GLAZED FIRE\u2013CLAY SANITARY APPLIANCES PART 1 GENERAL REQUIREMENTS 1. Scope \u2013 General requirements for materials, manufacture, finish, methods of test,",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "General requirements for materials, manufacture, finish, methods of test, sampling and inspection of all glazed fire-clay sanitary appliances.",
    },
    "IS 771 (PART 2): 1985": {
        "title":    "GLAZED FIRE \u2013 CLAY SANITARY APPLIANCES PART 2 SPECIFIC REQUIREMENTS OF KITCHEN AND LABORATORY SINKS 1. Scope \u2013 Lays down the pattern and sizes, constr",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Lays down the pattern and sizes, construction, dimensions and tolerances of kitchen and laboratory sinks made of fire-clay.",
    },
    "IS 771 (PART 5): 1979": {
        "title":    "GLAZED FIRE CLAY APPLIANCES, PART 5 SPECIFIC REQUIREMENTS OF SHOWER TRAYS For detailed information, refer to Specification for glazed fire-clay sanita",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Lays down the pattern, size, construction, dimensions, tolerances and finsh of shower trays made of fire-clay.",
    },
    "IS 771 (PART 7): 1981": {
        "title":    "GLAZED FIRE CLAY SANITARY APPLIANCES, PART 7 SPECIFIC REQUIREMENTS OF SLOP SINKS 1. Scope \u2013 Lays down the pattern, sizes, construction, dimensions, to",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Lays down the pattern, sizes, construction, dimensions, tolerances of slop sinks made of fire-clay.",
    },
    "IS 772: 1973": {
        "title":    "GENERAL REQUIREMENT FOR ENAMELLED CAST IRON SANITARY APPLIANCES 1. Scope \u2014 General requirement of material, thickness, warpage, enamelling, acid and a",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "General requirement of material, thickness, warpage, enamelling, acid and alkali resistance, inspection rules and marking, for enamelled cast iron sanitary appliance like water-closets and commodes.",
    },
    "IS 773: 1988": {
        "title":    "ENAMELLED CAST IRON WATER\u2013CLOSETS, RAILWAY COACHING STOCK TYPE 1. Scope\u2014Requirements for material, workmanship, manufacture, dimensions and finish of",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for material, workmanship, manufacture, dimensions and finish of enamelled cast iron-railway type water-closets generally used in the coaching stock of the Indian Railways.",
    },
    "IS 774: 2004": {
        "title":    "FLUSHING CISTERNS FOR WATER-CLOSETS AND URINALS (OTHER THAN PLASTIC CISTERNS) PART 1 GENERAL REQUIREMENTS 1. Scope \u2013 Requirements for manually-operate",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for manually-operated high- level and low-level flushing cisterns of capacities 5 litres and 10 litres , both single flush and dual flush types and 6/3 litres capacity duat flush cisterns, for water-closets, squatting pans and urinals, together with flush pipe details.",
    },
    "IS 1726: 1991": {
        "title":    "CAST IRON MANHOLE COVERS AND FRAMES Suitable for use in service lanes/roads, on pavements for use under medium-duty vehicular traffic including for ca",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Lays down basic and performance requirements for manhole covers and frames in cast- iron, intended for use in drainage and water works.",
    },
    "IS 2326: 1987": {
        "title":    "AUTOMATIC FLUSHING CISTERNS FOR URINALS (OTHER THAN PLASTIC CISTERNS) 1. Scope \u2013 Lays down the materials, nominal sizes, construction, performance req",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Lays down the materials, nominal sizes, construction, performance requirements and finish for automatic flushing cisterns of the type used for flushing urinals.",
    },
    "IS 2548 (PART 1): 1996": {
        "title":    "PLASTIC SEATS AND COVERS FOR WATER- CLOSETS, PART 1 THERMOSET SEATS AND COVERS * Phenolic moulding materials + Urea-formal dehyde, moulding material T",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Fifth Revision) * Phenolic moulding materials (third revision) + Urea-formal dehyde, moulding material (first revision) TABLE 1 DIMENSIONS OF SEATS AND COVERS All dimensions in millimetres. SL NO.",
    },
    "IS 2548 (PART 2): 1996": {
        "title":    "PLASTIC SEATS AND COVERS FOR WATER-CLOSETS, PART 2 \u2013 THERMOPLASTIC SEATS AND COVERS TABLE 1 DIMENSIONS OF SEATS AND COVERS ALL DIMENSIONS IN MILLIMETR",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 2 \u2013 THERMOPLASTIC SEATS AND COVERS (Fifth Revision) TABLE 1 DIMENSIONS OF SEATS AND COVERS ALL DIMENSIONS IN MILLIMETRES. SL NO. DESCRIPTION DIMENSION",
    },
    "IS 2556 (PART 1): 1994": {
        "title":    "VITREOUS SANITARY APPLIANCES (VITREOUS CHINA) PART 1 GENERAL REQUIREMENTS 1. Scope \u2013 General requirements relating to terminology, material and manufa",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "General requirements relating to terminology, material and manufacture, glazing, defects, minimum thickness, tolerance, performance and methods of test for vitreous sanitary appliances covered by various parts of the standard.",
    },
    "IS 2556 (PART 2): 1994": {
        "title":    "VITREOUS SANITARY APPLIANCES (VITREOUS CHINA) PART 2 - SPECIFIC REQUIREMENTS OF WASHDOWN WATER CLOSETS FIG. 1 PATERN 1 AND PATTERN 2 WATER CLOSETS",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "WASHDOWN WATER CLOSETS (Fourth Revision) FIG. 1 PATERN 1 AND PATTERN 2 WATER CLOSETS SP 21 : 2005",
    },
    "IS 2556 (PART 4): 1994": {
        "title":    "VITREOUS SANITARY APLLIANCES (VITREOUS CHINA) PART 4 SPECIFIC REQUIREMENTS OF WASH BASINS Note 2 \u2014 For general requirements refer to Part 1 General re",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 4 SPECIFIC REQUIREMENTS OF WASH BASINS (Third Revision) Note 2 \u2014 For general requirements refer to Part 1 General requirements of the standard. For detailed information, refer to IS 2556 (Part 4) : 1994 Specification for vitreous sanitary appliances (vitreous china): Part 4 Specific requirement",
    },
    "IS 2556 (PART 9): 1994": {
        "title":    "VITREOUS SANITARY APPLIANCES (VITREOUS CHINA ) PART 9 SPECIFIC REQUIREMENTS OFPEDASTAL TYPE BIDETS ii) Pattern 2 \u2014 Pedestal bidets without flushing ri",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 9 SPECIFIC REQUIREMENTS OFPEDASTAL TYPE BIDETS (Fourth Revision) ii) Pattern 2 \u2014 Pedestal bidets without flushing rim and over rim supply (See Fig. 2A & 2B). 3. Construction \u2014 Shall be of one piece construction.",
    },
    "IS 2556 (PART 14): 1994": {
        "title":    "VITREOUS SANITARY APPLIANCES (VITREOUS CHINA ) PART 14 SPECIFIC REQUIREMENTS OF INTEGRATED SQATTING PANS Note1 \u2014 For method of test, refer to 8 of the",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 14 SPECIFIC REQUIREMENTS OF INTEGRATED SQATTING PANS (First Revision) Note1 \u2014 For method of test, refer to 8 of the standard. Note 2 \u2014 For general requirements refer to Part 1 general requirements of the standard. For detailed information, refer to IS : 2556 (Part 14) : 1995 Specification for V",
    },
    "IS 5455: 1969": {
        "title":    "CAST IRON STEPS FOR MANHOLES 4. Tolerance \u2013 \u00b12 mm on all dimensions 5. Coating \u2013 Shall be coated with a material having tar base or with a black bitum",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "5. Coating \u2013 Shall be coated with a material having tar base or with a black bituminous composition or cashew-nut shell liquid. The coating shall be smooth and tenacious. It shall not flow when exposed to a temperature of temperature of 63\u00b0C and shall not brittle as to chip of at a temperature of o\u00b0",
    },
    "IS 5961: 1970": {
        "title":    "CAST IRON GRATING FOR DRAINAGE PURPOSES the frame \u00b1 2 mm. Note \u2014 For detailed dimensions see Fig. 1 of the standard. 5. Weight \u2013 75 kg minimum. 6. Coa",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Note \u2014 For detailed dimensions see Fig. 1 of the standard. 5. Weight \u2013 75 kg minimum. 6. Coating \u2013 Shall be with a material having tar base with a black bituminous composition. Coating shall be smooth and tenacious which will not flow at a temperature of 63\u00b0C, and which is not so brittle as to",
    },
    "IS 7231: 1994": {
        "title":    "PLASTIC FLUSHING CISTERNS FOR WATER CLOSETS AND URINALS T Talc as filler if used shall not exceed 20% Note\u2014 For materials of other components See Tabl",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Second Revision) T Talc as filler if used shall not exceed 20% Note\u2014 For materials of other components See Table1 of the standard. SP 21 : 2005",
    },
    "IS 11246: 1992": {
        "title":    "GLASS FIBRE REINFORCED POLYESTER RESIN (GRP) SQUATTING PANS * Glass fibre rovings for reinforcement of polyester and epoxide resin systems + Glass fib",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) * Glass fibre rovings for reinforcement of polyester and epoxide resin systems (first revision) + Glass fibre chopped strand mat for the reinforcement of epoxy, phenolic and polyester resin systems (first revision) 2. Material",
    },
    "IS 778: 1984": {
        "title":    "COPPER ALLOY GATE, GLOBE AND CHECK VALVES FOR WATER WORKS PURPOSES TABLE 1 MATERIALS FOR COMPONENT PARTS OF GATE, GLOBE AND CHECK VALVES SL. NO. COMPO",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Fourth Revision) TABLE 1 MATERIALS FOR COMPONENT PARTS OF GATE, GLOBE AND CHECK VALVES SL. NO. COMPONENT MATERIAL (1)",
    },
    "IS 781: 1984": {
        "title":    "CASTCOPPER ALLOYSCREW DOWN BIB TAPS ANDSTOPVALVES FOR WATERSERVICES 1. Scope \u2014 Requirements for copper alloy screw down bib taps and stop valves suita",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for copper alloy screw down bib taps and stop valves suitable for cold non- shock working pressure up to 1.0 MPa. Bib taps shall have screwed male inlet. Stop valves shall have screwed female end or male ends or mixed ends (mixed ends means one end screwed male and the other end screwed female).",
    },
    "IS 1701: 1960": {
        "title":    "MIXING VALVES FOR ABLUTIONARY AND DOMESTIC PURPOSE",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "MIXING VALVES FOR ABLUTIONARY AND DOMESTIC PURPOSE",
    },
    "IS 1703: 2000": {
        "title":    "WATER FITTINGS COPPER ALLOY FLOAT VALVES (HORIZONTAL PLUNGER TYPE) (Fourth Revison) For detailed information, refer to Specification for water fitting",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Fourth Revison) For detailed information, refer to IS 1703: 1999 Specification for water fittings copper alloy float Valves (horizontal plunger type) (fourth revision) SP 21 : 2005",
    },
    "IS 1711: 1984": {
        "title":    "SELF\u2013CLOSING TAPS FOR WATER SUPPLY PURPOSES For details information, refer to Specification for self-closing taps for water supplypurposes Endurance T",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Second Revision) For details information, refer to IS 1711 : 1984 Specification for self-closing taps for water supplypurposes (second revision) 6.2 Endurance Test \u2014 Shall not show any leakage or failure of the spring or other working parts after 50,000 operations",
    },
    "IS 1795: 1982": {
        "title":    "PILLAR TAPS FOR WATER SUPPLY PURPOSES 6. Finish \u2014 Shall be nickel-chromium plated. Shall be capable of taking high polish. 7. Testing \u2014 Shall withstan",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "6. Finish \u2014 Shall be nickel-chromium plated. Shall be capable of taking high polish. 7. Testing \u2014 Shall withstand internally applied hydraulic pressure of 2 MPa (20Kgf/cm2) for 2 minutes",
    },
    "IS 2692: 1989": {
        "title":    "FERRULES FOR WATER SERVICES 1. Scope \u2013 Lays down nominal sizes and requirements regarding material, manufactrue and workmanship, construction, samplin",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Lays down nominal sizes and requirements regarding material, manufactrue and workmanship, construction, sampling and testing of copper alloy screw- down ferrules for use on water supply mains.",
    },
    "IS 3004: 1979": {
        "title":    "PLUG COCKS FOR WATER SUPPLY PURPOSES nuts, union nuts and tail pipes. Taper of the side of plug and body shall be 1 in 15 (1 in 7/5 included angle). T",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements of plug cocks of 15 mm, 20 mm and 25 mm nominal size with a key head for use underground for water supply purposes up to 1 MPa working pressure.",
    },
    "IS 3311: 1979": {
        "title":    "WASTE PLUG AND ITS ACCESSORIES FOR SINKS AND WASH-BASINS. For detailed information, refer to Specification for waste plug and its accessories for sink",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) For detailed information, refer to IS 3311 : 1979 Specification for waste plug and its accessories for sinks and wash-basins (first revision). * Glazed fireclay sanitary appliances, Part 2 Specific requirements for kitchen and laboratorysinks (second revision) t Vitreous sanitary ap",
    },
    "IS 4346: 1982": {
        "title":    "WASHERS FOR USE WITH FITTINGS FOR WATER SERVICES",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "WASHERS FOR USE WITH FITTINGS FOR WATER SERVICES",
    },
    "IS 5312 (PART 1): 2004": {
        "title":    "SWING CHECK TYPE REFLUX (NON-RETURN) VALVES FOR WATER WORKS PURPOSES PART 1 SINGLE - DOOR PATTERN 1. Scope \u2014 Requirements for flanged reflux valves of",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for flanged reflux valves of single door, swing check type used for water works purposes of sizes 50 to 600 mm.",
    },
    "IS 5312 (PART 2): 1986": {
        "title":    "SWING CHECK TYPE REFLUX (NON-RETURN) VALVES FOR WATER WORKS PURPOSES PART 2 MULTI - DOOR PATTERN Note\u2014 For alternative material see Table 1of the stan",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "PART 2 MULTI - DOOR PATTERN (First Revision) Note\u2014 For alternative material see Table 1of the standard. 5. Design and Manufacture 5.1 Body may be made in two parts-inlet shell (having",
    },
    "IS 8931: 1993": {
        "title":    "COPPER ALLOY FANCY SINGLE TAPS COMBINATION TAP ASSEMBLY AND STOP VALVES FOR WATER SERVICES",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "COMBINATION TAP ASSEMBLY AND STOP VALVES FOR WATER",
    },
    "IS 9338: 1984": {
        "title":    "CAST IRON SCREW-DOWN STOP VALVES AND STOP AND CHECK VALVES FOR WATER WORKS PURPOSES 1. Scope \u2013 Requirements for flanged cast iron screw- down stop val",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements for flanged cast iron screw- down stop valves from 15 to 300 mm nominal sizes of the following types used for water supply up to 450 C : Globe stop valve; Angle stop valve; oblique stop valve; Globe stop and check value and Angle stop and check valves.",
    },
    "IS 9739: 1981": {
        "title":    "PRESSURE REDUCING VALVES FOR DOMESTIC WATER SUPPLY SYSTEM Screen of the strainer shall have a minimum unobstructed open flow area (total area of holes",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(Third Revision) 4.3 Screen of the strainer shall have a minimum unobstructed open flow area (total area of holes ) equal to or greater than twice the nominal pipe flow area. The maximum hole demension of the screen shall not exceed 1/12 of the valve orifice escape diameter.",
    },
    "IS 9758: 1981": {
        "title":    "FLUSH VALVES AND FITTINGS FOR WATER CLOSETS AND URINAL 4. Manufacture and Construction Flush valve of nominal sizes 15, 25 and 32mm shall have an outl",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "4. Manufacture and Construction 4.1 Flush valve of nominal sizes 15, 25 and 32mm shall have an outlet of 20, 32 and 40mm outside diameter respectively. 4.2 Fush valve shallbe self closing and non-concussive",
    },
    "IS 9762: 1994": {
        "title":    "POLYETHYLENE FLOATS (SPHERICAL) FOR FLOAT VALVES + High density polyehylene materials for moulding and extrusion. 4. Dimensions and Tolerances \u2014 See T",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "(First Revision) + High density polyehylene materials for moulding and extrusion. (First Revision) 4. Dimensions and Tolerances \u2014 See Table 1. 5.",
    },
    "IS 9763: 2000": {
        "title":    "PLASTIC BIB TAPS, PILLAR TAPS, ANGLE VALVES FOR HOT AND COLD WATER SERVICES (Second Revison) 1. Scope \u2014 Requirements regarding material, dimensions, c",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Requirements regarding material, dimensions, construction finish, and testing of plastic bib taps, pillar taps, stop valve & angle valves for hot and cold water services.",
    },
    "IS 12234: 1988": {
        "title":    "PLASTIC EQUILIBRIUM FLOAT VALVES FOR COLD WATER SERVICES Note \u2014 For method of test refer to Appendices A to C of the standard. For detailed informatio",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Note \u2014 For method of test refer to Appendices A to C of the standard. For detailed information, refer to IS 12234 : 1988 Specification for plastic equilibrium float valve for cold water services. SP 21 : 2005",
    },
    "IS 13049: 1991": {
        "title":    "DIAPHRAGM TYPE (PLASTIC BODY) FLOAT OPERATED VALVES FOR COLD WATER SERVICES Note\u2014 For method of test refer to Appendices Aot C of the standard. For de",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Note\u2014 For method of test refer to Appendices Aot C of the standard. For detailed information, refer to IS 13049 : 1991 Specification for diaphragm type (plastric body) float operated valves for cold water services. 0 025 .0",
    },
    "IS 13114: 1991": {
        "title":    "FORGED BRASS GATE, GLOBE AND CHECK VALVES FOR WATER WORKS PURPOSES c) Check valves: Swing type and Lift type 5. Dimensions and Tolerances \u2013 See Tables",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "c) Check valves: Swing type and Lift type 5. Dimensions and Tolerances \u2013 See Tables 2 & 3 6. Design and Manufacture",
    },
    "IS 13349: 1992": {
        "title":    "SINGLE FACED CAST IRON THIMBLE MOUNTED SLUICE GATES 5. Materials \u2013 see Table 3 TABLE 3 MATERIALS Sl. No. Item Material",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "5. Materials \u2013 see Table 3 TABLE 3 MATERIALS Sl. No. Item Material",
    },
    "IS 14846: 2000": {
        "title":    "SLUICE VALVES FOR WATER WORKS PURPOSES (50 TO 1200 mm SIZE)",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "SLUICE VALVES FOR WATER WORKS PURPOSES (50 TO 1200 mm SIZE)",
    },
    "IS 779: 1994": {
        "title":    "WATER METERS (DOMESTIC TYPE) (Sixth Revision) Note: For material details, see Annex. B of the standard.",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Note: For material details, see Annex. B of the standard. SP 21 : 2005",
    },
    "IS 2104: 1981": {
        "title":    ". WATER METER BOXES (DOMESTIC TYPE) * Water meters (domestic type) (sixth revision)",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "* Water meters (domestic type) (sixth revision) SP 21 : 2005",
    },
    "IS 2373: 1981": {
        "title":    "WATER METERS (BULK TYPE) Capacity Ratings for Intermediate flows Nominal Capacity Ratings of watermeters Size in Litres per hour mm Vane-Wheel Type He",
        "category": "Sanitary Appliances and Water Fittings",
        "section":  10,
        "scope":    "Covers bulk type water meters of the following types : a) Vane-wheel (impeller) type water meters from 50 to 300 mm ; and b) Helical type water meters from 50 to 500 mm",
    },
    "IS 204 (PART 1): 1991": {
        "title":    "TOWER BOLTS PART 1 FERROUS METALS 1. Scope \u2013 Requirements for tower bolts made of ferrous metals. 2. Types",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for tower bolts made of ferrous metals.",
    },
    "IS 205: 1992": {
        "title":    "NON FERROUS METAL BUTT HINGES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "NON FERROUS METAL BUTT HINGES 1. Scope \u2013 Requirements for mild steel Tee and strap hinges that are commonly used in generall building construction. 2. Types Tee hinges shall be of the following types\u2014 Type Designation 1 Light weight 2 Medium weight 3 Heavy weight Strap hinges shall be of the following types \u2014 Type Designation 1 Light weight strap 2 Medium weight 3 Heavy weight 3. Materials i) Mild",
    },
    "IS 206: 1992": {
        "title":    "TEE AND STRAP HINGES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "TEE AND STRAP HINGES 1. Scope \u2013 Requirements for materials, manufacture, dimensions and finish of door handles of the type that are commonly fixed to doors. 2. Types Type 1 Cast Type 2 Pressed oval Type 3 Pressed half oval Type 4 Fabricated 3. Materials Type 1 Cast iron, malleable cast iron, cast brass, cast aluminium or zinc and 3 alloydiecasting, Type 2and 3 Mild steel, and Type 4 Brass or alumi",
    },
    "IS 208: 1996": {
        "title":    "DOOR HANDLES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "DOOR HANDLES 1. Scope \u2013 Requirements regarding materials, dimensions, manufacture and finish of mild steel sliding door bolts commonly used in general building constrcution for locking doors, gates, etc, with padlocks. 2. Types i) Plate Type, and ii) Clip or bolt type. 3. Sizes (a) Plate type sliding bolts\u2014 150, 200, 250, 300, 375 and 450 mm; and (b)Clip or bolt type sliding bolts\u2014 200, 250, 300,",
    },
    "IS 281: 1991": {
        "title":    "MILD STEEL SLIDING DOOR BOLTS FOR USE WITH PADLOCKS",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "MILD STEEL SLIDING DOOR BOLTS FOR USE WITH PADLOCKS",
    },
    "IS 362: 1991": {
        "title":    "PARLIAMENT HINGES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "PARLIAMENT HINGES 1. Scope \u2013 Requirements regarding materials, manufacture, dimensions, manufacture and finish of hasps and staples. 2. Types Type Description 1. Mild steel, brass or aluminium alloy hasps and staples\u2014safety type. 2. Mild steel hasps and staples\u2014 wire type. 3. Sizes Mild Steel Hasps and Staples Type 1\u2014 90, 115, 150 and 175 mm. Brass or Aluminium Alloy Hasps and Staples Type 1 \u2013 90,",
    },
    "IS 363: 1993": {
        "title":    "HASPS AND STAPLES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "HASPS AND STAPLES SUMMARY OF FAN LIGHT CATCH 1. Scope\u2014Requirements regarding material, dimensions, manufacture and finish of fan light catches commonly used on ventilators in buildings. 2. Types a) Mild steel fan light catches, b) Aluminium alloy fan light catches,and c) Cast brass fan light catches. 3. Materials a) Mild steel sheet shall satisfy prescribed bend test. b) Mild steel wire shall have",
    },
    "IS 452: 1973": {
        "title":    "DOOR SPRING RAT- TAIL TYPE",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "DOOR SPRING RAT- TAIL TYPE SUMMARY OF DOUBLE-ACTING SPRING HINGES 1. Scope \u2013 Requirements for material, dimensions manufacture, finish and tests of double-acting spring hinges and corresponding blank hinges used generally for swing doors. 2. Types a) Mild Steel double-acting spring hinges, and b) Brass double-acting spring hinges. 3. Sizes Size of Spring Size of Blank Hinge Hinge mm mm 100 70 125",
    },
    "IS 1019: 1974": {
        "title":    "RIM LATCHES 1. Scope \u2013 Requirements regarding material, dimensions, manufacture and finish of rim latches for general use. 2. Handling of Rim Latches\u2014",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements regarding material, dimensions, manufacture and finish of rim latches for general use.",
    },
    "IS 1341: 1992": {
        "title":    "STEEL BUTT HINGES (Sixth Revision) 1. Scope\u2014Requirements regarding material, dimensions, manufacture and finish of mild steel butt hinges. 2. Types",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements regarding material, dimensions, manufacture and finish of mild steel butt hinges.",
    },
    "IS 1823: 1980": {
        "title":    "FLOOR DOOR STOPPERS 1. Scope \u2013 Requirements for floor door stopper suitable for use with door shutters of 30, 35, 40, and 45 mm thickness. 2. Material",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for floor door stopper suitable for use with door shutters of 30, 35, 40, and 45 mm thickness.",
    },
    "IS 2209: 1976": {
        "title":    "MORTICE LOCKS (VERTICAL TYPE) 1. Scope \u2013 Requirements for mortice locks (vertical type) 2. Sizes \u2013 65, 75 and 100 mm. Size shall be denoted by the ove",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for mortice locks (vertical type)",
    },
    "IS 2681: 1993": {
        "title":    "NON-FERROUS METAL SLIDING DOOR BOLTS (ALDROPS) FOR USE WITH PADLOCKS",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "(ALDROPS) FOR USE WITH PADLOCKS",
    },
    "IS 3818: 1992": {
        "title":    "CONTINUOUS (PIANO) HINGES 4. Requirements a) Knuckles shall be straight and at right angle to the flap. b) Hinge pin shall be of mild steel in the cas",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "4. Requirements a) Knuckles shall be straight and at right angle to the flap. b) Hinge pin shall be of mild steel in the case of mild steel hinges and shall be of mild steel",
    },
    "IS 3828: 1966": {
        "title":    "VENTILATOR CHAINS",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "VENTILATOR CHAINS 1. Scope \u2013 Covers types and the requirements regarding materials, dimensions, manufacture and finish of steel back flap hinges. 2. Types \u2013 a) Light weight hinges, and b) Heavy weight hinges 3. Materials \u2013 a) Flap Steelcover plate b) Pin Mild Steel wire For detailed information, refer to Specification for steel back flap hinges ( second revision). 4. Sizes Light Weight Hinges \u2014 20",
    },
    "IS 3843: 1995": {
        "title":    "STEEL BACK FLAP HINGES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "STEEL BACK FLAP HINGES 1. Scope \u2013 Requirements for mortice night latches for general use. 2. General \u2014 Nominal size shall be denoted by overall length of the body measured from the outside face of the fore end to the rear end. Termed \u2018Left hand\u2019 if fitted on \u2018Left hand door\u2019 and \u2018Right Hand\u2019 if fitted in \u2018Right Hand door\u2019. Two lever latches and latches with more than two livers shall have non-inte",
    },
    "IS 3847: 1992": {
        "title":    "MORTICE NIGHT LATCHES Case plate, face plate ii) Aluminium alloy sheet and striking plate iii) Cast brass (copper content shall not be less than 60 pe",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Case plate, face plate ii) Aluminium alloy sheet and striking plate iii) Cast brass (copper content shall not be less than 60 percent",
    },
    "IS 4621: 1975": {
        "title":    "INDICATING BOLT FOR USE IN PUBLIC BATHS AND LAVATORIES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "INDICATING BOLT FOR USE IN PUBLIC BATHS AND LAVATORIES",
    },
    "IS 5187: 1972": {
        "title":    "FLUSH BOLTS",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Specification for FLUSH BOLTS. Covers materials, dimensions, physical and mechanical requirements.",
    },
    "IS 5899: 1970": {
        "title":    "BATH ROOM LATCHES 1. Scope \u2013 Requirements for material, size and finish of bathroom latches. 2. Shape and Size Overall size \u2014 40 \u00d7 50 mm Thickness \u2013 1",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for material, size and finish of bathroom latches.",
    },
    "IS 5930: 1970": {
        "title":    "MORTICE LATCH (VERTICAL TYPE) 1. Scope \u2013 Requirements for mortice latches for use on doors, such as bath room doors, W.C. doors and doors to private r",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for mortice latches for use on doors, such as bath room doors, W.C. doors and doors to private rooms.",
    },
    "IS 6315: 1992": {
        "title":    "FLOOR SPRINGS (HYDRAULICALLY REGULATED) FOR HEAVY DOORS 1. Scope \u2013 Requuirements for concealed type floor springs (hydraulically regulated) for vertic",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requuirements for concealed type floor springs (hydraulically regulated) for vertical doors weighing not more than 125 kg. In case of doors consisting of more than one leaf the weight of each leaf shall not exceed 125 kg.",
    },
    "IS 6318: 1971": {
        "title":    "PLASTIC WINDOW STAYS AND FASTENERS 1. Scope \u2013 Lays down performance and functional requirements of window stays made of polypropylene and fasteners (h",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Lays down performance and functional requirements of window stays made of polypropylene and fasteners (handles) made of nylon.",
    },
    "IS 6343: 1982": {
        "title":    "DOOR CLOSERS (PNEUMATICALLY REGULATED) FOR LIGHT DOORS WEIGHING UPTO 40 KG 1. Scope\u2013Requirements for door closers",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for door closers (pneumatically regulated) for use on light doors weighing up to 40 kg.",
    },
    "IS 6607: 1972": {
        "title":    "REBATED MORTICE LOCKS (VERTICAL TYPE) 1. Scope \u2013 Requirements for rebated mortice locks suitable for use on double leaf doors with rebated meeting sti",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for rebated mortice locks suitable for use on double leaf doors with rebated meeting stiles.",
    },
    "IS 7196: 1974": {
        "title":    "HOLD FAST . 1. Scope \u2014 Requirements for mild steel hold fasts for use with wooden doors and window frames. 2. Size and Dimensions Shall be as given in",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for mild steel hold fasts for use with wooden doors and window frames.",
    },
    "IS 7197: 1974": {
        "title":    "DOUBLE ACTION FLOOR SPRINGS (WITHOUT OIL CHECK) FOR HEAVY DOORS 1. Scope \u2013 Requirements for concealed type floor springs (without oil check) for verti",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements for concealed type floor springs (without oil check) for vertical doors weighing not more than 125 kg. For doors having more than one leaf, the weight of each leaf shall not exceed 125 kg.",
    },
    "IS 7534: 1985": {
        "title":    "SLIDING LOCKING BOLTS FOR USE WITH PADLOCKS \u00b1 \u00b1 \u00b1 \u00b1 \u00b1",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "SLIDING LOCKING BOLTS FOR USE WITH PADLOCKS \u00b1 \u00b1 \u00b1 \u00b1 \u00b1 \u00b1 \u00b1 \u00b1 \u00b1 \u00b1 \u00b1 \u00b1 \u00b1",
    },
    "IS 7540: 1974": {
        "title":    "MORTICE DEAD LOCKS",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "MORTICE DEAD LOCKS 1. Scope \u2013 Requirements for mortice sliding door locks having lever mechanism. 2. Sizes \u2013 30 mm, 50 mm, 70 mm and 100 mm. Size shall be denoted by the overall length of the body in millimetres measured from the outside face of the fore\u2013end to rear end. Measured length shall not vary by more than 3 mm from the length specified for size. 3. Shape and Design \u2013 Any shape but shall b",
    },
    "IS 8760: 1978": {
        "title":    "MORTICE SLIDING DOOR LOCKS WITH LEVER MECHANISM",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "MORTICE SLIDING DOOR LOCKS WITH LEVER MECHANISM",
    },
    "IS 9106: 1979": {
        "title":    "RISING BUTT HINGES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "RISING BUTT HINGES 1. Scope \u2013 Requirements for materials, construction, dimensions and finish of rim locks of two types commonly fixed to single and double-leaf doors in buildings. 2. Types Type 1 \u2013 Left hand or right-hand, and Type 2 \u2013 Reversible 3. Size \u2013 100 and 150 mm The size of the rim lock shall be denoted by the length of face over the body in millimetres. The measured length shall not var",
    },
    "IS 9131: 1979": {
        "title":    "RIM LOCKS",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "RIM LOCKS 1. Scope \u2013 Requirements regarding materials, dimensions, manufacture and finish of stays and fasteners made of mild steel of the types that are commonly used in windows. 2. Types a) Type 1 \u2013 Mild steel stays and fasterners having tabular cross section, and b) Type 2 \u2013 Mild steel stays and fasterners made out of one piece sheet. 3. Materials i) Mild steel sheets ii) Mild steel bars Note\u2014",
    },
    "IS 10019: 1981": {
        "title":    "MILD STEEL STAYS AND FASTENERS",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "MILD STEEL STAYS AND FASTENERS 1. Scope \u2013 Requirements regarding materials, dimensions, manufacture and finish of numericals. 2. Materials i) Cast brass ii) Cast bronze iii) Cast aluminium Note\u2014 For details of materials see 2 and Table 1 of the standard. 3. Sizes \u2014 25, 50, 75, 100, 150 and 300 mm. The thickness of the numericals shall not be less than 2 mm, and the width shall be as agreed upon be",
    },
    "IS 10342: 1982": {
        "title":    "CURTAIN RAIL SYSTEM For detailed information, refer to Specification for curtain rail system. 1. Scope \u2013 Requirements regarding materials, manufacture",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "Requirements regarding materials, manufacture, dimensions, testing and finish of rails, runners and hooks used in the curtain rail system.",
    },
    "IS 12817: 1997": {
        "title":    "STAINLESS STEEL BUTT HINGES",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "STAINLESS STEEL BUTT HINGES 1. Scope \u2013 Covers the dimensions and requirements for PVC handrail covers for use on metal strip handrails. 2. Material \u2013 Handrails covers are manufactured by extrusion using plasticized PVC compound of desired formulation and colour. 3. Sizes \u2013 PVC handrail covers are normally made available in widths to match the desired width of metal TABLE 1 - REQUIREMENTS OF PVTC H",
    },
    "IS 12867: 1989": {
        "title":    "PVC HAND RAIL COVERS",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "PVC HAND RAIL COVERS 1. Scope \u2013 Requirement for concealed type hydraulically operated door closers, fixed in concealed position within the thickness of the panel on vertical, hinge type doors opening to one side only and not weiging more than 80 kg. This standard does not cover overhead type door closers covered in * or pneumatic type door closers or closers working on only mechanical device. 2. N",
    },
    "IS 14912: 2001": {
        "title":    "DOOR CLOSERS - CONCEALED TYPE (HYDRAULICALLY REGULATED) \u00b1",
        "category": "Builders Hardware",
        "section":  11,
        "scope":    "DOOR CLOSERS - CONCEALED TYPE (HYDRAULICALLY REGULATED) \u00b1",
    },
    "IS 12049: 1987": {
        "title":    "DIMENSIONS AND TOLERANCES RELATING TO WOOD BASED PANEL MATERIALS",
        "category": "Wood Products",
        "section":  12,
        "scope":    "DIMENSIONS AND TOLERANCES RELATING TO WOOD BASED PANEL MATERIALS",
    },
    "IS 849: 1994": {
        "title":    "COLD SETTING CASE IN GLUE FOR WOOD 1. Scope \u2013 Requirements for cold setting casein glue used in the wood panel industry, wood-work and joinery industr",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements for cold setting casein glue used in the wood panel industry, wood-work and joinery industry.",
    },
    "IS 851: 1978": {
        "title":    "SYNTHETIC RESIN ADHESIVES FOR CONSTRUCTION WORK (NON-STRUCTURAL) IN WOOD Type Symbol Gap -Filling Close Contact Adhesive Adhesive Boiling Water Proof",
        "category": "Wood Products",
        "section":  12,
        "scope":    "(First Revision) Type Symbol Gap -Filling Close Contact Adhesive Adhesive Boiling Water Proof",
    },
    "IS 852: 1994": {
        "title":    "ANIMAL GLUE FOR GENERAL WOOD-WORKING PURPOSES 1. Scope \u2013 Requirements of animal glue for general wood-working purposes. 2. Material \u2014The glue shall be",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements of animal glue for general wood-working purposes.",
    },
    "IS 1328: 1996": {
        "title":    "VENEERED DECORATIVE PLYWOOD 1. Scope \u2013 Covers types of plywood with ornamental veneers on one or both faces used for decorative purposes, such as furn",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements for veneered decorative plywood with or without overlap, used for decorative purposes in furniture and interior woodwork.",
    },
    "IS 4990: 1993": {
        "title":    "PLYWOOD FOR CONCRETE SHUTTERING WORK 1. Scope \u2013 Requirements of plywood for concrete shuttering and form work. 2. Types \u2013 Plywood for concrete shutter",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements of plywood for use as shuttering material in concrete formwork. Covers bonding quality, dimensions, tolerances and face veneer requirements.",
    },
    "IS 5509: 2000": {
        "title":    "FIRE RETARDANT PLYWOOD 1. Scope \u2013 Covers the fire retardant chemicals, method of treatment, retentions and requirements of fire retardant plywood. 2.",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Covers the fire retardant chemicals, method of treatment, retentions and requirements of fire retardant plywood.",
    },
    "IS 5539: 1969": {
        "title":    "PRESERVATIVE TREATED PLYWOOD 1. Scope \u2013 Treatment of plywood for protection against fungi, termites and other insects and marine borers and requiremme",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Treatment of plywood for protection against fungi, termites and other insects and marine borers and requiremments of preservatives treated plywood.",
    },
    "IS 7316: 1974": {
        "title":    "DECORATIVE PLYWOOD USING PLURALITY OF VENEER FOR DECORATIVE FACES 1. Scope \u2013 Covers decorative plywood with ornamental faces produced by use of plural",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Covers decorative plywood with ornamental faces produced by use of plurality of veneers meant for decorative use, such as interior panelling of buildings, buses, ships, etc. and for decorative furniture of all types.",
    },
    "IS 10701: 1983": {
        "title":    "STRUCTURAL PLYWOOD 1. Scope \u2013 Requirements of plywood for structural purposes, such as stressed skin panels, plywood web beams, sheathing, silos, rail",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements of plywood for structural purposes, such as stressed skin panels, plywood web beams, sheathing, silos, rail and ship containers.",
    },
    "IS 13957: 1994": {
        "title":    "METAL FACED PLYWOOD 1. Scope \u2013 Covers manufacture and requirements of metal faced plywood composite. The scope is limited to the use of galvanized iro",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Covers manufacture and requirements of metal faced plywood composite. The scope is limited to the use of galvanized iron sheet or aluminium sheet only, as metal sheet.",
    },
    "IS 1658: 1977": {
        "title":    "FIBRE HARD BOARDS 1. Scope \u2013 Requirements of fibre hardboards for general purposes. This standard does not cover requirements of insulation boards, wo",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements of fibre hardboards for general purposes. This standard does not cover requirements of insulation boards, wood particle boards (chip boards), and similar boards.",
    },
    "IS 1659: 2004": {
        "title":    "BLOCK BOARDS 1. Scope \u2013 Essential requirements of commercial and decorative blockboards meant for interior and exterior uses. 2. Grades and Types",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Essential requirements of commercial and decorative blockboards meant for interior and exterior uses.",
    },
    "IS 3087: 1985": {
        "title":    "WOOD PARTICLE BOARDS (MEDIUM DENSITY) FOR GENERAL PURPOSES 1. Scope\u2013 Requirements of medium density wood particle boards for general purposes, having",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements of medium density wood particle boards for general purposes, having specific grativity in the range 0.5 to 0.9. This standard does not cover veneered particle boards, moulded particle boards, high and low density particle boards or particle boards faced by impregnated paper surfaces.",
    },
    "IS 3097: 1980": {
        "title":    "VENEERED PARTICLE BOARDS 1. Scope \u2013 Requirements, such as, grades and types, material, manufacture, dimensions and tests for veneered particle boards.",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements, such as, grades and types, material, manufacture, dimensions and tests for veneered particle boards.",
    },
    "IS 3129: 1985": {
        "title":    "LOW DENSITY PARTICLE BOARDS 1. Scope \u2013 Essential requirments of low density particle boards having specific grativity not exceeding 2. Material Timber",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Essential requirments of low density particle boards having specific grativity not exceeding 0.4",
    },
    "IS 3308: 1981": {
        "title":    "WOOD WOOL BUILDING SLABS Note\u2014 For test procedures, refer to Appendix B of the standard, Method for the determination of thermal conductivity of therm",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Note\u2014 For test procedures, refer to Appendix B of the standard, IS 3346 : 1980 Method for the determination of thermal conductivity of thermal insulation materials (Two slab, guarded hot-plate method) (first revision) and IS 8225 : 1987 Method of measurement of sound absorption in a reverberation ro",
    },
    "IS 3478: 1966": {
        "title":    "HIGH DENSITY WOOD PARTICLE BOARDS Note \u2014For test procedures, refer to various parts of Methods of test for wood particle boards and boards from other",
        "category": "Wood Products",
        "section":  12,
        "scope":    "other lignocellulosic materials ( first revision ) and 9.3 of the standard. For detailed information, refer to IS 3478 :1966 Specification for high density wood particle board",
    },
    "IS 12406: 2003": {
        "title":    "MEDIUM DENSITY FIBRE BOARDS FOR GENERAL PURPOSES 1. Scope \u2014 Requirements of medium density fibre boards for general purposes having density in the ran",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements of medium density fibre boards for general purposes having density in the range of 600 \u2013 900 kg/m3. This standard does not cover veneered or laminated or other specially treated boards, moulded boards, etc.",
    },
    "IS 12823: 1990": {
        "title":    "WOOD PRODUCTS-PRELAMINATED PARTICLES BOARDS Impregnated Overlay\u2014An absorbent tissue paper having a weight of 18-40 g/m2 impregnated in a suitable synt",
        "category": "Wood Products",
        "section":  12,
        "scope":    "3.3 Impregnated Overlay\u2014An absorbent tissue paper having a weight of 18-40 g/m2 impregnated in a suitable synthetic resin and dried to a volatile content of 4-8 percent. 4. Finish \u2014The finish of the paper overlaid board",
    },
    "IS 14276: 1995": {
        "title":    "CEMENT BONDED PARTICLE BOARDS 1. Scope \u2013 Requirements of cement bonded wood particle boards.This standard does not cover particle boards bonded with s",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Requirements of cement bonded wood particle boards.This standard does not cover particle boards bonded with synthetic resin adhesives.",
    },
    "IS 14587: 1998": {
        "title":    "PRE-LAMINATED MEDIUM DENSITY FIBRE BOARD TABLE 1 PHYSICAL AND MECHANICAL PROPERTIES PROPERTY REQUIREMENT Grade I Grade II",
        "category": "Wood Products",
        "section":  12,
        "scope":    "TABLE 1 PHYSICAL AND MECHANICAL PROPERTIES PROPERTY REQUIREMENT Grade I Grade II 1.1",
    },
    "IS 14616: 1999": {
        "title":    "LAMINATED VENEER LUMBER 1. Scope\u2014Covers laminated veneer lumber (LVL) of density range to in which most natural structural wood fall. Its applications",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Covers laminated veneer lumber (LVL) of density range 0.6 to 0.75 in which most natural structural wood fall. Its applications include all the end uses to which structural wood has been traditionally used, such as, beams, rafters, stringers, joists, posts and framework construction, stiles, rails and frames of doors and windows, vehicle bodies, railways coaches, containers, framework of furniture, cabinets, shelving etc.",
    },
    "IS 13958: 1994": {
        "title":    "BAMBOO MAT BOARD FOR GENERAL PURPOSES 1. Scope \u2013 Covers the method of manufacture and the requirements of bamboo mat board used for general purposes.",
        "category": "Wood Products",
        "section":  12,
        "scope":    "Covers the method of manufacture and the requirements of bamboo mat board used for general purposes.",
    },
    "IS 1003 (PART 1): 2003": {
        "title":    "TIMBER PANELLED AND GLAZED SHUTTERS PART 1 DOOR SHUTTERS 1. Scope \u2014 Requirements regarding material, sizes, construction, workmanship, finish, inspect",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding material, sizes, construction, workmanship, finish, inspection and testing of timber door shutters with timber, plywood, blockboard, veneered particle board, asbestos cement sheet, wire guage and glass panels used in domestic buildings, offices, schools, hospitals, etc. The shutters could be single panelled or multipanelled with or without glazing. This standard does not cover timber door shutters for industrial and other s",
    },
    "IS 1003 (PART 2): 1994": {
        "title":    "TIMBER PANELLED AND GLAZED SHUTTERS PART 2 \u2013 WINDOW AND VENTILATOR SHUTTERS",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "PART 2 \u2013 WINDOW AND VENTILATOR SHUTTERS (Third Revision)",
    },
    "IS 2191 (PART 1): 1983": {
        "title":    "WOODEN FLUSH DOOR SHUTTERS (CELLULAR AND HOLLOW CORE TYPE) PART 1 PLYWOOD FACE PANELS 1. Scope \u2013 Requirements regarding types, sizes, material, constr",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding types, sizes, material, construction, workmanship and finish, and tests of cellular and hollow core wooden flush door shutters with face panels of plywood or cross-band and face veneers.",
    },
    "IS 2191 (PART 2): 1983": {
        "title":    "WOODEN FLUSH DOOR SHUTTERS (CELLULAR AND HOLLOW CORE TYPE) PART 2 \u2013 PARTICLE BOARD AND HARDBOARD FACE PANELS 1. Scope \u2013 Requirements regarding materia",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding material, grade, types, sizes, construction, finishes and tests of wooden flush door shutters of cellular and hollow core type with particle board face panels (both veneered and unveneered) and hard board face panels.",
    },
    "IS 2202 (PART 1): 1999": {
        "title":    "WOODEN FLUSH DOOR SHUTTERS (SOLID CORE TYPE) PART 1 PLYWOOD FACE PANELS (Sixth Revision) 1. Scope \u2013 Requirements regarding types, sizes, material, con",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding types, sizes, material, consttruction, workmanship and finish, and tests, of solid core wooden flush door shutters with face panels of plywood or cross-band and face veneers.",
    },
    "IS 2202 (PART 2): 1983": {
        "title":    "WOODEN FLUSH DOOR SHUTTERS (SOLID CORE TYPE) PART 2 PARTICLE BOARD AND HARDBOARD FACE PANELS For detailed information, refer to Specification for wood",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "PART 2 PARTICLE BOARD AND HARDBOARD FACE PANELS (Third Revision) For detailed information, refer to IS 2202 (Part 2) :1983 Specification for wooden flush door shutters (solid core type) : Part 2 Particle board and har board face panels (third revision). 2.1 0",
    },
    "IS 4021: 1995": {
        "title":    "TIMBER DOOR, WINDOW AND VENTILATOR FRAMES 1. Scope \u2013 Requirements regarding material, construction, workmanship and sizes of timber door, window and v",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding material, construction, workmanship and sizes of timber door, window and ventilator frames generally used in residential and institutional buildings. 1.1 This standard does not cover timber door, window and ventilator frames for commercial, industrial and other special buildings, such as, workshops and garages.",
    },
    "IS 4962: 1968": {
        "title":    "WOODEN SIDE SLIDING DOORS For detailed information, refer to Specification for wooden side sliding doors. 1. Scope \u2014 Requirements regarding material,",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding material, type, shape fabrication, assembly and finish of wooden side sliding doors (of the straight sliding type), its gear components and fittings.",
    },
    "IS 6198: 1992": {
        "title":    "LEDGED, BRACED AND BATTENED TIMBER DOOR SHUTTERS 1. Scope \u2013 Requirements regarding material, sizes, construction, workmanship and finish of ledged, br",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding material, sizes, construction, workmanship and finish of ledged, braced and battened timber door shutters.",
    },
    "IS 15380: 2003": {
        "title":    "MOULDED RAISED HIGH DENSITY FIBRE (HDF) PANEL DOORS 1. Scope \u2013 This standard lays down requirements regarding types, sizes, material, construction, wo",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "This standard lays down requirements regarding types, sizes, material, construction, workmanship and finish and tests for high density fibre (HDF) panel doors.",
    },
    "IS 1038: 1983": {
        "title":    "STEEL DOOR, WINDOWS AND VENTILATORS * Symbolic designation of directions of closing and faces of doors, windows and shutters. 10 HF 6 / 10 HF 6 10 HS",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "* Symbolic designation of directions of closing and faces of doors, windows and shutters. 10 HF 6 / 10 HF 6 10 HS 12 / 10 HS 12",
    },
    "IS 1361: 1978": {
        "title":    "STEEL WINDOWS FOR INDUSTRIAL BUILDINGS b) Ventilator (Opening part of a Sash) \u2013 Shall be of one size and designed to fit into outer frame of IN 10 C 1",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Deals with steel windows suitable for use in industrial buildings and designed to suit openings based on a module of 10 cm.",
    },
    "IS 1948: 1961": {
        "title":    "ALUMINIUM DOORS, WINDOWS AND VENTILATORS",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "ALUMINIUM DOORS, WINDOWS AND VENTILATORS",
    },
    "IS 1949: 1961": {
        "title":    "ALUMINIUM WINDOWS FOR INDUSTRIAL BUILDINGS 1. Scope \u2013 Deals with aluminium windows suitable for use in industrial buildings and designed to suit openi",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Deals with aluminium windows suitable for use in industrial buildings and designed to suit openings based on a module of 10 cm.",
    },
    "IS 4351: 2003": {
        "title":    "STEEL DOOR FRAMES 1. Scope \u2013 Requirements regarding material, dimensions and construction of steel door frames for internal and external use. 2. Mater",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding material, dimensions and construction of steel door frames for internal and external use.",
    },
    "IS 6248: 1979": {
        "title":    "METAL ROLLING SHUTTERS AND ROLLING GRILLS 1. Scope \u2013 Requirements regarding mateials, fabrication and finish of metal rolling shutters and rolling gri",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding mateials, fabrication and finish of metal rolling shutters and rolling grills for normal use.",
    },
    "IS 7452: 1990": {
        "title":    "HOT ROLLED STEEL SECTIONS FOR DOORS, WINDOWS AND VENTILATORS 1. Scope \u2013 Requirements regarding materials, nominal dimension and mass, dimensional and",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding materials, nominal dimension and mass, dimensional and mass tolerances, surface finish and packing for hot rolled steel sections used for doors, windows, ventilators and sashes.",
    },
    "IS 10451: 1983": {
        "title":    "STEEL SLIDING SHUTTERS (TOP HUNG TYPE) 1. Scope \u2013 Requirements regarding materials, type, shape, fabrication, assembly and finish of the top hung stee",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding materials, type, shape, fabrication, assembly and finish of the top hung steel sliding shutters.",
    },
    "IS 10521: 1983": {
        "title":    "COLLAPSIBLE GATES 1. Scope \u2013 Requirements regarding materials, fabrication and finish of different types of collapsible gates. 2. Types a)",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding materials, fabrication and finish of different types of collapsible gates.",
    },
    "IS 14856: 2000": {
        "title":    "GLASS FIBRE REINFORCED PLASTIC (GRP) PANEL TYPE DOOR SHUTTERS FOR INTERNAL USE 1. Scope \u2013 Requirements regarding types, sizes, material, construction,",
        "category": "Doors, Windows and Shutters",
        "section":  13,
        "scope":    "Requirements regarding types, sizes, material, construction, workmanship, finish, performance requirements and sampling of fibre glass reinforced plastic door shutters for use in residential and industrial building.",
    },
    "IS 432 (PART 1): 1982": {
        "title":    "MILD STEEL AND MEDIUM TENSILE STEEL BARS AND HARD-DRAWN STEEL WIRE FOR CONCRETE REINFORCEMENT PART 1 MILD STEEL AND MEDIUM TENSILE STEEL BARS .2 Ovali",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "CONCRETE REINFORCEMENT PART 1 MILD STEEL AND MEDIUM TENSILE STEEL BARS (Third Revision) 5.1.2 Ovality and out-of-square\u2014 Permissible ovality for round bars and out-of-square of square bars shall be 75 percent of total tolerance (plus and minus)",
    },
    "IS 1566: 1982": {
        "title":    "HARD - DRAWN STEEL WIRE FABRIC FOR CON- CRETE REINFORCEMENT 1. Scope \u2013 Requirements for hard-drawn steel wire fabric consisting of hard-drawn steel wi",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "Requirements for hard-drawn steel wire fabric consisting of hard-drawn steel with cross wires electrically welded to them for use as concrete reinforcement",
    },
    "IS 1785 (PART 2): 1983": {
        "title":    "PLAIN HARD- DRAWNSTEEL WIRE FOR PRESTRESSED CONCRETE. RART 2 -AS DRAWN WIRE 1. Scope \u2013 Requirements for manufacture supply and testing of plain \u2018as-dr",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "Requirements for manufacture supply and testing of plain \u2018as-drawn\u2019 steel wire for use in prestressed concrete pipes and similar other purposes.",
    },
    "IS 1786: 1985": {
        "title":    "PLAIN HIGH STRENGTH DEFORMED STEEL BARS AND WIRES FOR CONCRETE REINFORCEMENT +75 \u201325 +50 \u20130",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "AND WIRES FOR CONCRETE REINFORCEMENT",
    },
    "IS 2090: 1983": {
        "title":    "HIGH TENSILE STEEL BARS USED IN PRESTRESSED CONCRETE",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "HIGH TENSILE STEEL BARS USED IN PRESTRESSED CONCRETE",
    },
    "IS 6003: 1983": {
        "title":    "INDENTED WIRE FOR PRESTRESSED CONCRETE Note \u2013 For test procedures, refer to 7 of the standard and Mechanical testing of metals-Tensile testing For det",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "Note \u2013 For test procedures, refer to 7 of the standard and IS 1608 : 1995 Mechanical testing of metals-Tensile testing (second revision) For detailed information, refer to IS 6003 : 1983 Specification for intended wire for prestressed concrete (first revision).",
    },
    "IS 6006: 1983": {
        "title":    "UNCOATED STRESS RELIEVED STRAND FOR PRESTRESSED CONCRETE",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "UNCOATED STRESS RELIEVED STRAND FOR PRESTRESSED CONCRETE",
    },
    "IS 7887: 1992": {
        "title":    "MILD STEEL WIRE ROD FOR GENERAL ENGINEERING PURPOSES 3. Condition of Material on Delivery \u2013 The hot-rolled wire rod shall be supplied in the form of c",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "(First Revision) 3. Condition of Material on Delivery \u2013 The hot-rolled wire rod shall be supplied in the form of coils or straigtened and cut lengths.The size and weight of coils shall be as agreed.",
    },
    "IS 13620: 1993": {
        "title":    "FUSION BONDED EPOXY COATED REINFORCING BARS",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "FUSION BONDED EPOXY COATED REINFORCING BARS",
    },
    "IS 14268: 1995": {
        "title":    "UNCOATED STRESS RELIEVED LOW RELAXATION SEVEN- PLY STRAND FOR PRESTRESSED CONCRETE . TABLE 1 PHYSICAL PROPERTIES Class Nominal Breaking % Proof Load",
        "category": "Concrete Reinforcement",
        "section":  14,
        "scope":    "CONCRETE . TABLE 1 PHYSICAL PROPERTIES Class Nominal Breaking 0.2% Proof Load",
    },
    "IS 1977: 1996": {
        "title":    "LOW TENSILE STRUCTURAL STEEL 5. Tensile Properties: See Table 2 TABLE 2 TENSILE PROPERTIES Grade Tensile Yield Percent Internal Desig",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "Requirements for low tensile structural steel for general structural purposes where low tensile properties are required. Used in structures where economy in weight is not a criterion.",
    },
    "IS 2062: 1999": {
        "title":    "Steel for General Structural Purposes",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "Requirements for steel for general structural purposes used as plates, strips, shapes, sections and bars for structural purposes. Covers grades E 165, E 250, E 300, E 350, E 410 and E 450.",
    },
    "IS 8500: 1991": {
        "title":    "STRUCTURAL STEEL- MICRO ALLOYED (MEDIUM AND HIGH STRENGTH QUALITIES)",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "(MEDIUM AND HIGH STRENGTH QUALITIES)",
    },
    "IS 11587: 1986": {
        "title":    "STRUCTURAL WEATHER RESISTANCE STEELS 1. Scope \u2013 Requirements for high strength low alloy weather resistant structural steels in the form of plates, st",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "Requirements for high strength low alloy weather resistant structural steels in the form of plates, strips, sections and bars for welded, riveted or bolted construction requiring atmospheric corrosion resistance.",
    },
    "IS 277: 2003": {
        "title":    "GALVANIZED STEEL SHEETS (PLAIN AND CORRUGATED) (Sixth Revision) TABLE 2 MASS OF COATING Grade of Minimum Average Minimum Coating Coating",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "(Sixth Revision) TABLE 2 MASS OF COATING Grade of Minimum Average Minimum Coating Coating",
    },
    "IS 412: 1975": {
        "title":    "EXPANDED METAL STEEL SHEETS FOR GENERAL PURPOSES Ref. No. Size of Mesh Largest Standard Size of Sheet (Nominal)`",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "(Second Revision) Ref. No. Size of Mesh Largest Standard Size of Sheet (Nominal)`",
    },
    "IS 513: 1994": {
        "title":    "COLD ROLLED LOW CARBON STEEL SHEETS AND STRIPS 1. Scope \u2013 Requirements of cold rolled low carbon steel sheets and strips for bending and drawing purpo",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "Requirements of cold rolled low carbon steel sheets and strips for bending and drawing purpose and where the surface is of prime importance. It covers sheets and strips up to 4 mm thick both in coil form and cut lengths.",
    },
    "IS 1079: 1994": {
        "title":    "HOT ROLLED CARBON STEEL SHEETS AND STRIPS 1. Scope\u2014 Requirements of Hot rolled carbon steel sheets including pack rolled sheets and strips intended fo",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "Requirements of Hot rolled carbon steel sheets including pack rolled sheets and strips intended for cold forming, drawing and general engineering purposes.",
    },
    "IS 3502: 1994": {
        "title":    "STEEL CHEQUERED PLATES + Steel for general structural purposes . ++ Structural steel (ordinary quality) .",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "Requirements for steel chequered plates used for flooring, stair treads and other structural purposes where anti-slip surface is required.",
    },
    "IS 7226: 1974": {
        "title":    "COLD\u2013ROLLED MEDIUM, HIGH CARBON AND LOW ALLOY STEEL STRIP FOR GENERAL ENGINEERING PURPOSES For detailed information, refer to Specification for cold-r",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "For detailed information, refer to IS 7226 : 1974 Specification for cold-rolled medium, high carbon and low alloy steel strip for general engineering purposes. 6. Surface Finish \u2013 Bright Finish. Note \u2013 For rolling tolerances see 9 of the standard. * Methods for rockwell hardness tes for metallic mat",
    },
    "IS 12313: 1988": {
        "title":    "HOT\u2013DIP TERNE COATED CARBON STEEL SHEETS TABLE 1 CHEMICAL COMPOSITION (LADLE ANALYSIS) Quality C Mn P S Max",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "TABLE 1 CHEMICAL COMPOSITION (LADLE ANALYSIS) Quality C Mn P S Max",
    },
    "IS 1148: 1982": {
        "title":    "HOT ROLLED STEEL RIVETS BARS (UPTO 40 MM DIAMETER) FOR STRUCTURAL PURPOSES (Third Revision ) 1. Scope \u2013 Requirement for hot-rolled steel rivet bars in",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "Requirement for hot-rolled steel rivet bars in size up to 40 mm diameter used for the manufacture of hot forged rivets for structural purposes.",
    },
    "IS 1149: 1982": {
        "title":    "HIGH TENSILE STEEL RIVETS BARS FOR STRUCTURAL PURPOSES (Third Revision ) 1. Scope \u2013 Requirement for high tensile steel rivet bars in size up to 40 mm",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "Requirement for high tensile steel rivet bars in size up to 40 mm diameter for structural purposes.",
    },
    "IS 1161: 1998": {
        "title":    "STEEL TUBES FOR STRUCTURAL PURPOSES. \u20138 percent 1) Single tube light Medium \u00b1 10 percent Heavy 2) 10 tonne lots light",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "\u20138 percent 1) Single tube light Medium \u00b1 10 percent Heavy 2) 10 tonne lots light",
    },
    "IS 4923: 1997": {
        "title":    "HOLLOW STEEL SECTIONS FOR STRUCTURAL USE TABLE 1 DIMENSIONS AND PROPERTIES OF SQUARE HOLLOW SECTIONS Designation Depth Thick",
        "category": "Structural Steels",
        "section":  15,
        "scope":    "(Second Revision) TABLE 1 DIMENSIONS AND PROPERTIES OF SQUARE HOLLOW SECTIONS Designation Depth Thick-",
    },
    "IS 733: 1983": {
        "title":    "WROUGHT ALUMINIUM AND ALUMINIUM ALLOY BARS, RODS AND SECTIONS FOR GENERAL ENGINEERING PURPOSES 1. Scope \u2013 Requirements for wrought aluminium and alumi",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "Requirements for wrought aluminium and aluminium alloy bars, rods and sections for general engineering purposes.",
    },
    "IS 736: 1986": {
        "title":    "WROUGHT ALUMINIUM AND ALUMINIUM ALLOY PLATE FOR GENERAL ENGINEERING PURPOSES Designation Typical uses vessels, irrigation tubing, heat exchangers, ute",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "(Third Revision) Designation Typical uses vessels, irrigation tubing, heat exchangers, utensils and pressure cookers,",
    },
    "IS 737: 1986": {
        "title":    "WROUGHT ALUMINIUM AND ALUMINIUM ALLOY SHEET AND STRIP FOR GENERAL ENGINEERING PURPOSES Designation Typical uses 24345 Heavy duty forgings, structures",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "(Third Revision) Designation Typical uses 24345 Heavy duty forgings, structures where high mechanical peoperties are",
    },
    "IS 738: 1994": {
        "title":    "WROUGHT ALUMINIUM AND ITS ALLOYS-DRAWN TUBES FOR GENERAL ENGINEERING PURPOSES Designation Typical uses 31000 General purpose alloy for moderate streng",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "(Third Revision) Designation Typical uses 31000 General purpose alloy for moderate strength applications,",
    },
    "IS 739: 1992": {
        "title":    "WROUGHT ALUMINIUM AND ALUMINIUM ALLOYS-WIRE FOR GENERAL ENGINEERING PURPOSES Designation Typical Uses 43000 Filler wires for brazing and soldering, we",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "(Third Revision) Designation Typical Uses 43000 Filler wires for brazing and soldering, welding rods, sprays gun wires.",
    },
    "IS 740: 1977": {
        "title":    "WROUGHT ALUMINIUM ALLOY RIVET STOCK FOR GENERAL ENGINEERING PURPOSES 1. Scope \u2013 Requirements for wrought aluminium and aluminium alloys rivet stock fo",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "Requirements for wrought aluminium and aluminium alloys rivet stock for general engineering purposes.",
    },
    "IS 1254: 1991": {
        "title":    "CORRUGATED ALUMINIUM SHEET 1. Scope \u2013 Material, profile, dimensions and finish for corrugated aluminium sheets meant for following uses: a) General pu",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "Material, profile, dimensions and finish for corrugated aluminium sheets meant for following uses: a) General purpose, b) Industrial, and c) Building.",
    },
    "IS 1285: 2002": {
        "title":    "WROUGHT ALUMINIUM AND ALUMINIUM ALLOY EXTRUDED ROUND TUBE AND HOLLOW SECTIONS FOR GENERAL ENGINEERING PURPOSES 1. Scope \u2013 Requirements of extruded rou",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "Requirements of extruded round tube and hollow sections made from wrought aluminium and aluminium alloys for general engineering purposes.",
    },
    "IS 2525: 1982": {
        "title":    "DIMENSION FOR WROUGHT ALUMINIUM AND ALUMINIUM ALLOYS, WIRE 1. Scope \u2013 Lays down dimensions and tolerances for wrought aluminium alloys in the form of",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "Lays down dimensions and tolerances for wrought aluminium alloys in the form of wire.",
    },
    "IS 2676: 1981": {
        "title":    "DIMENSIONS FOR WROUGHT ALUMINIUM AND ALUMINIUM ALLOYS, SHEET AND STRIP 1. Scope \u2013 Lays down dimensions and tolerances for wrought aluminium alloys, sh",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "Lays down dimensions and tolerances for wrought aluminium alloys, sheet and strip.",
    },
    "IS 2677: 1979": {
        "title":    "DIMENSIONS FOR WROUGHT ALUMINIUM AND ALLOYS PLATES AND HOT-ROLLED SHEETS 1. Scope \u2013 Lays down the dimensions and tolerances for wrought aluminium and",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "Lays down the dimensions and tolerances for wrought aluminium and aluminium alloys, plate and hot-rolled sheets.",
    },
    "IS 2678: 1987": {
        "title":    "DIMENSIONS AND TOLERANCE FOR WROUGHT ALUMINIUM AND ALUMINIUM ALLOYS DRAWN ROUND TUBES TABLE 1 DIMENSIONS OF DRAWN ROUND TUBE WITH PARALLEL BORE All di",
        "category": "Light Metals and Their Alloys",
        "section":  16,
        "scope":    "Lays down the dimensions and tolerances for wrought aluminium and aluminium alloy drawn round tube with parallel bore. } } } }",
    },
    "IS 3908: 1986": {
        "title":    "ALUMINIUM EQUAL LEG ANGLES 1. Scope \u2013 Cover the material, dimensions and sectional properties of aluminium equal leg angles for structural use and oth",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Cover the material, dimensions and sectional properties of aluminium equal leg angles for structural use and other applications.",
    },
    "IS 3909: 1986": {
        "title":    "ALUMINIUM UNEQUAL LEG ANGLES of longer and shorter legs and thickness of the section in mm. For example Alu 80 \u00d7 60 \u00d7 6 3. Dimensions",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "of longer and shorter legs and thickness of the section in mm. For example Alu 80 \u00d7 60 \u00d7 6 3. Dimensions \u2013",
    },
    "IS 3921: 1985": {
        "title":    "ALUMINIUM CHANNELS For detailed information, refer to Specifications for aluminium channels . ALC 40 \u00d7 20 - ALC 40 \u00d7 20 - ALC 50 \u00d7 30 - ALC 50 \u00d7 30 -",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "For detailed information, refer to IS 3921 : 1985 Specifications for aluminium channels (first revision). ALC 40 \u00d7 20 - 0.63 ALC 40 \u00d7 20 - 0.44 ALC 50 \u00d7 30 - 1.55 ALC 50 \u00d7 30 - 0.88 ALC 50 \u00d7 30 - 1.14",
    },
    "IS 5384: 1985": {
        "title":    "ALUMINIUM I\u2013BEAM 1. Scope \u2013 Covers the material, dimensions and sectional properties of aluminium I- beam sections for structural use and other applic",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Covers the material, dimensions and sectional properties of aluminium I- beam sections for structural use and other applications.",
    },
    "IS 6445: 1985": {
        "title":    "ALUMINIUM TEE - SECTIONS 4. Dimensions Designation ALT 25 \u00d7 25 - ALT 100 \u00d7 75 - ALT 30 \u00d7 30",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "4. Dimensions 4.1 Designation ALT 25 \u00d7 25 - 0.4 ALT 100 \u00d7 75 - 5.4 ALT 30 \u00d7 30 - 0.5",
    },
    "IS 808: 1989": {
        "title":    "DIMENSIONS FOR HOT ROLLED STEELBEAM, COLUMN, CHANNEL AND ANGLE SECTIONS 1. Scope \u2013 Covers the nominal dimensions, and sectional properties of hot roll",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Covers the nominal dimensions, and sectional properties of hot rolled sloping flange beam and column sections, sloping and parallel flange channel sections and equal and unequal leg angle sections.",
    },
    "IS 811: 1987": {
        "title":    "COLD FORMED LIGHT GAUGE STRUCTURAL STEEL SECTIONS 1. Scope \u2013 Lays down dimensions mass, sectional properties and requirements for corrosion protection",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Lays down dimensions mass, sectional properties and requirements for corrosion protection for cold formed light guage open wall steel sections for structural and other general applications, having minimum thickness of 1.25mm.",
    },
    "IS 1173: 1978": {
        "title":    "HOT ROLLED AND SLIT STEEL TEE BARS 1. Scope \u2013 Lays down nominal dimensions, weight and basic geometrical properties. 2. Classification \u2013 a)",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Lays down nominal dimensions, weight and basic geometrical properties.",
    },
    "IS 1730: 1989": {
        "title":    "STEEL PLATES SHEETS STRIPS AND FLATS, FOR STRUCTURAL AND GENERAL ENGINEERING PURPOSES 1. Scope \u2013 Specifies nominal dimensions, nominal mass and surfac",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Specifies nominal dimensions, nominal mass and surface area (for sheets) of hot-rolled steel plates, sheets, strips and flats for structural and general engineering purposes.",
    },
    "IS 1732: 1989": {
        "title":    "STEEL BARS, ROUND AND SQUARE FOR STRUCTURAL AND GENERAL ENGIINEERING PURPOSES-DIMENSIONS 1. Scope \u2013 Specifies dimensions, sectional areas and mass of",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Specifies dimensions, sectional areas and mass of hot-rolled round and square steel bars for structural and general engineering purposes. This standard does not cover bars for rivets and threaded components.",
    },
    "IS 1863: 1979": {
        "title":    "ROLLED STEEL BULB FLATS (First Revision ) 1. Scope \u2013 Specifies dimenions, sectional properties and dimensional tolerances of hot-rolled steel bulb fla",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Specifies dimenions, sectional properties and dimensional tolerances of hot-rolled steel bulb flats",
    },
    "IS 2314: 1986": {
        "title":    "STEEL SHEET PILLING SECTIONS 1. Scope \u2013 Stipulates dimensions and dimensional tolerances for Z-type , U-type an flat-type profile of hot rolled steel",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Stipulates dimensions and dimensional tolerances for Z-type , U-type an flat-type profile of hot rolled steel sheet piling sections. Sectional properties of these sections as calculated with the nominal dimensions are also included.",
    },
    "IS 3443: 1980": {
        "title":    "CRANE RAIL SECTION 6. Dimensions and Properties Desig- Cross- Weight Bottom nation Sec- (kg/m)",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "6. Dimensions and Properties Desig- Cross- Weight Bottom nation Sec- (kg/m)",
    },
    "IS 3954: 1991": {
        "title":    "HOT ROLLED CHANNEL SECTIONS FOR GENERAL ENGINEERING PURPOSES \u2013 DIMENSIONS",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "ENGINEERING PURPOSES \u2013 DIMENSIONS",
    },
    "IS 3964: 1980": {
        "title":    "LIGHT RAILS 1. Scope \u2014 Requirements of light rail sections. 2 . Designation \u2014 By letters ISLR followed by a figure denoting weight in kg per metre of",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Requirements of light rail sections.",
    },
    "IS 8081: 1976": {
        "title":    "SLOTTED SECTIONS For detailed information, refer to Specifications for slotted sections 7. Tolerance of Dimensions \u2013 Flange Sectional Dimensions \u2013 The",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "7. Tolerance of Dimensions \u2013 7.1 Flange Sectional Dimensions \u2013 The tolerance on sum of the dimensions of all flanges shall not exceed the following: Nominal Size",
    },
    "IS 12778: 2004": {
        "title":    "HOT ROLLED PARALLEL FLANGE STEEL SECTION FOR BEAMS, COLUMNS AND BEARING PILES DIMENSIONS AND SECTION PROPERTIES 1 Scope \u2013 Covers the nominal dimension",
        "category": "Structural Shapes",
        "section":  17,
        "scope":    "Covers the nominal dimensions, mass and sectional properties of hot rolled parallel flange beams, columns and bearing piles.",
    },
    "IS 814: 2004": {
        "title":    "COVERED ELECTRODES FOR MANUAL METAL ARC WELDING OF CARBON AND CARBON MANGANESE STEEL (Sixth Revision) 1. Scope \u2013 Requirements for covered carbon and c",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "Requirements for covered carbon and carbon manganese steel electrodes for carbon and carbon manganese steel, including hydrogen controlled electrodes for manual metal arc welding of mild and medium tensile steels including structural steels, depositing weld metal having a tensile strength not more than 610 MPa. 1.1 Electrodes designed specifically for repair welding, often markedted in India as \u2018low heat input\u2019 electrodes are not covered in this ",
    },
    "IS 1278: 1972": {
        "title":    "FILLER RODS FOR GAS WELDING 1. Scope \u2013 Requirements of ferrous and non-ferrous filler rods for gas welding made of the following materials supplied in",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "Requirements of ferrous and non-ferrous filler rods for gas welding made of the following materials supplied in cut lengths. a) Structural steels, b) Austenitic stainless steels, c) Cast irons (excluding spheroidal graphite and malleable iron castings), d) Copper and copper alloys, e) Nickel and nickel alloys, f) Aluminium and aluminium alloys, and g) Magnesium and magnesium alloys.",
    },
    "IS 1395: 1982": {
        "title":    "LOW AND MEDIUM ALLOY STEEL COVERED ELECTRODES FOR MANUAL METAL ARC WELDING 1. Scope \u2013 Covers the requirements for low and medium alloy steel covered e",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "Covers the requirements for low and medium alloy steel covered electrodes for manual metal arc welding.",
    },
    "IS 4972: 1968": {
        "title":    "RESISTANCE SPOT \u2013 WELDING ELECTRODES 1. Scope \u2013 Code numbers (in metric units), dimensional requirements, and physical and mechanical properties for a",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "Code numbers (in metric units), dimensional requirements, and physical and mechanical properties for a series of spot-welding electrodes, cap electrodes and shanks, mainly intented for resistance spot welding of ferrous and non-ferrous metals. This standard covers . electrodes with standard ISO tapers and with Morse tapers.",
    },
    "IS 5511: 1991": {
        "title":    "COVERED ELECTRODES FOR MANUAL METAL ARC WELDING OF CAST IRON 1. Scope \u2013 Specifies a system of classification and coding and covers requirements for co",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "Specifies a system of classification and coding and covers requirements for covered electrodes for manual metal arc welding of cast iron.",
    },
    "IS 5897: 1985": {
        "title":    "ALUMINIUM AND ALUMINIUM ALLOY WELDING RODS AND WIRES AND MAGNESIUM ALLOY WELDING RODS",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "RODS AND WIRES AND MAGNESIUM ALLOY WELDING RODS",
    },
    "IS 5898: 1970": {
        "title":    "COPPER AND COPPER ALLOY BARE SOLID WELDING RODS AND ELECTRODES",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "COPPER AND COPPER ALLOY BARE SOLID WELDING RODS AND ELECTRODES",
    },
    "IS 6560: 1996": {
        "title":    "MOLYBDENUM AND CHROMIUM-MOLYBDENUM LOW ALLOY STEEL WELDING RODS AND BARE ELECTRODES FOR GAS SHIELDED ARC WELDING 1. Scope Requirements of solid filler",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "1.1 Requirements of solid filler rods and wires for welding. It covers molybdenum and chromium molybdenum low alloy steel rods and wires for use in inert-gas tungsten arc welding (TIG), gas metal arc welding (MIG) or CO2 welding processes. The chemical composition and tensile properties of filler rods and wires are also specified. 1.2 This standard also specifies the mechanical properties of the weld deposits.",
    },
    "IS 7280: 1974": {
        "title":    "BARE WIRE ELECTRODES FOR SUBMERGED ARC WELDING OF STRUCTURAL STEELS 1. Scope \u2013 Requirements of solid filler wires for submerged arc welding of structu",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "Requirements of solid filler wires for submerged arc welding of structural steels (28-50 kgf/mm2 yield strength and 34-70 kgf/mm2 ultimate tensile strength).",
    },
    "IS 8363: 1976": {
        "title":    "BARE WIRE ELECTRODES FOR ELECTROSLAG WELDING OF STEELS 1. Scope \u2013 Requirements of solid bare wire electrodes for electroslag welding of carbon and low",
        "category": "Welding Electrodes and Wires",
        "section":  18,
        "scope":    "Requirements of solid bare wire electrodes for electroslag welding of carbon and low alloy steels. Note. \u2013 This standard is intended to serve as a guide for the manufacturer and selection of bare wire electrodes for electroslag welding of carbon manganese and low alloy steels.",
    },
    "IS 207: 1964": {
        "title":    "GATE AND SHUTTER HOOKS AND EYES (Revised) 1. Scope \u2013 Requirements for gate and shutter hooks and eyes which are commonly used on doors and windows for",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for gate and shutter hooks and eyes which are commonly used on doors and windows for keeping them in position when kept open.",
    },
    "IS 723: 1972": {
        "title":    "STEEL COUNTER SUNK HEAD WIRE NAILS 1. Scope \u2013 Requirements of steel countersunk head wire nails. 2. Dimensions and Tolerances (in mm) Dimensions",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements of steel countersunk head wire nails.",
    },
    "IS 724: 1964": {
        "title":    "MILD STEEL AND BRASS CUP, RULER AND SQUARE HOOKS AND SCREW EYES (Revised) 1. Scope \u2013 Requirements for mild steel and brass cup, ruler and square hooks",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for mild steel and brass cup, ruler and square hooks and screw eyes.",
    },
    "IS 725: 1961": {
        "title":    "COPPER WIRE NAILS (Revised) 1. Scope \u2013 Covers the following types of copper wire nails: a) Rose-head boat nails, square shank, square point. b) Counte",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Covers the following types of copper wire nails: a) Rose-head boat nails, square shank, square point. b) Countersunk-head boat nails, square shank, sharp square point. c) Countersunk-head boat nails, square shank, round point, d) Wrought tacks e) Cut-lath nails (Cut tacks)",
    },
    "IS 730: 1978": {
        "title":    "HOOK BOLTS FOR CORRUGATED SHEET ROOFING",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "HOOK BOLTS FOR CORRUGATED SHEET ROOFING",
    },
    "IS 1120: 1975": {
        "title":    "COACH SCREW 3. Designation \u2014 As an example, a hexagon head coach screw of screw No. 10, length 30 mm and made of steel, shall be designated as Coach S",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "3. Designation \u2014 As an example, a hexagon head coach screw of screw No. 10, length 30 mm and made of steel, shall be designated as Coach Screw No.10 x 30 IS 1120 Steel. Note \u2013 In regard to the requirements not covered in the standard, refer to IS 451 : 1999 Technical supply conditions for wood",
    },
    "IS 1363 (PART 1): 2002": {
        "title":    "ISO 4016 : 1999 HEXAGON HEAD BOLTS, SCREWS AND NUTS OF PRODUCT GRADE C PART 1 : HEXAGON HEAD BOLTS (SIZE RANGE M5 TO M64) Prefered threads \u2013 M5, M6 an",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "PART 1 : HEXAGON HEAD BOLTS (SIZE RANGE M5 TO M64) (Fourth Revision) 2.1 Prefered threads \u2013 M5, M6 and M64 2.2 Non - Prfered threads \u2013 M14, M18, M22, M27, M33, M39, M45, M52, and M60 3. Specifications \u2013 See Table 1",
    },
    "IS 1363 (PART 2): 2002": {
        "title":    "ISO 4018 : 1999 HEXAGON HEAD BOLTS, SCREWS & NUTS OF PRODUCT GRADE C PART 2 : HEXAGON HEAD SCREWS (SIZE RANGE M5 TO M64) 4. Designation \u2013 Example for",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "PART 2 : HEXAGON HEAD SCREWS (SIZE RANGE M5 TO M64) (Fourth Revision) 4. Designation \u2013 Example for the designation of a hexagon head screw with thread M12, nominal length Note \u2013 For corresponding Indian standards of cerain International standard referred, along with their degree of equivalence, refe",
    },
    "IS 1363 (PART 3): 2002": {
        "title":    "ISO 4034 : 1999 HEXAGON HEAD BOLTS, SCREWS & NUTS OF PRODUCT GRADE C PART 3 : HEXAGON NUTS (SIZE RANGE M5 TO M64) Note \u2014For details, of preferred and",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "PART 3 : HEXAGON NUTS (SIZE RANGE M5 TO M64) (Fourth Revision) Note \u2014For details, of preferred and non preferred size refer to Tables 1 and 2 of the Standard. 2.1 Preferred threads 2.2 Non \u2013 preferred threads \u2013 M14, M18, M22, M27,",
    },
    "IS 1364 (PART 1): 2002": {
        "title":    "ISO 4014 : 1999 HEXAGON HEAD BOLTS, SCREWS AND NUTS OF PRODUCT GRADES A AND B PART 1 : HEXAGON HEAD BOLTS (SIZE RANGE M1.6 TO M64) For detailed inform",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "AND B PART 1 : HEXAGON HEAD BOLTS (SIZE RANGE M1.6 TO M64) (Third Revision) For detailed information, refer to IS 1364 (Part 1) :1992. Specification for ISO 4014 : 1998. Hexagon head bolts, screws and nuts of product grades A and B \u2013 Part I \u2013 Hexagon head bolts (size range M1.6 to M64) (third revisi",
    },
    "IS 1364 (PART 2): 2002": {
        "title":    "ISO 4017 : 1999 HEXAGON HEAD BOLTS, SCREWS & NUTS OF PRODUCT GRADE A & B PART 2 : HEXAGON HEAD SCREWS (SIZE RANGE TO M64) 1. Scope \u2013 Gives specificati",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Gives specifications for hexagon head screws with threads from M1.6 upto and including M64, of product grade A for threads M1 6 to M24 and nominal lengths up to and including 10 d or 150 mm, whichever is shorter and product grade B for threads over M24 or nominal lengths over 10 d or 150 mm, whichever is shorter.",
    },
    "IS 1364 (PART 3): 2002": {
        "title":    "ISO 4032 : 1999 HEXAGON HEAD BOLTS, SCREWS AND NUTS OF PRODUCT GRADES A & B PART 3 : HEXAGON NUTS (SIZE RANGE M1.6 TO M64) For detailed information, r",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "PART 3 : HEXAGON NUTS (SIZE RANGE M1.6 TO M64) (Fourth Revision) For detailed information, refer to IS 1364(PART 3) :1992.Specification for ISO 4032:1986 Hexagon head bolts,screw and nuts of product grades A and B\u2014 Part 3\u2014 Hexagon nuts (Size range M1.6 to M64) (third revision). 4.",
    },
    "IS 1364 (PART 4): 2003": {
        "title":    "ISO 4035 : 1999 HEXAGON HEAD BOLTS,SCREWS&NUTS OFPRODUCTGRADEA &B PART 4 HEXAGON THIN NUTS (CHAMFERED) (SIZE RANGE M1.6 TO M64) Note \u2013 For correspondi",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "PART 4 HEXAGON THIN NUTS (CHAMFERED) (SIZE RANGE M1.6 TO M64) (Fourth Revision) Note \u2013 For corresponding Indian standards of certain International standards referred, along with their degree of equivalence, refer to National Foreward of the standard. For detailed information, refer to IS 1364 (Part",
    },
    "IS 1364 (PART 5): 2002": {
        "title":    "ISO 4036 : 1999 HEXAGON HEAD BOLTS, SCREWS AND NUTS OF PRODUCT GRADES A AND B PART 5 HEXAGON THIN NUTS (UNCHAMFERED) (SIZE RANGE M1.6 TO M10) 1. Scope",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Gives specifications for hexagon thin nuts with metric dimensions and thread diameters from 1.6, up to and including 10 mm and product grade B.",
    },
    "IS 1365: 1978": {
        "title":    "SLOTTED COUNTERSUNK HEAD SCREWS 1. Scope \u2013 Requirements for slotted countersunk head screws in the diameter range 1 to 20 mm. 2. Requirements Mechanic",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for slotted countersunk head screws in the diameter range 1 to 20 mm.",
    },
    "IS 1366: 2002": {
        "title":    "ISO 1207 - 1992 SLOTTED CHEESE HEAD SCREWS 2. Dimensions \u2013 M1.6, M2, M2.5, M3, (M3.5), M4, M6, M8, M12 3. Specification \u2013 See Table 1 4. Designation \u2013",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "(Third Revision) 2. Dimensions \u2013 M1.6, M2, M2.5, M3, (M3.5), M4, M6, M8, M12 3. Specification \u2013 See Table 1 4. Designation \u2013 as per sheet attahced.",
    },
    "IS 1929: 1982": {
        "title":    "HOT FORGED STEEL RIVETS FOR HOT CLOSING (12 TO 36 mm DIAMETER) 1. Scope \u2013 Requirements of hot forged solid mild steel and high tensile steel rivets (s",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements of hot forged solid mild steel and high tensile steel rivets (snap head, flat countersunk head and flat head rivets) for hot closing in the diameter range 12 to 36 mm intended for general engineering purposes.",
    },
    "IS 2016: 1967": {
        "title":    "PLAIN WASHERS",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "PLAIN WASHERS Punched Washers, Type A, for Hexagonal Bolts and Screws (Diameter External Thickness For Bolt/ Size of Hole) Diameter Screw 4 M1.6 () 5 (M1.8) 5 M2 () 6 (M2.2) M2.5 7 M3 (4) 8 (M3.5) 9 M4 (5) 10 1 (M4.5) 10 1 M5 M6 () 14 (M7) 9 17 M8 11 21 2 M10 14 24 M12 (16) 28 (M14) 18 30 M16 (20) 34 (M18) 22 37 M20 (24) 39 (M22) 26 44 4 M24 (30) 50 4 (M27) 33 56 4 M30 (36) 60 5 (M33) 39 66 5 M36",
    },
    "IS 2155: 1982": {
        "title":    "COLD FORGED SOLID STEEL RIVETS FOR HOT CLOSING (6 TO 16 mm DIAMETER) 20 0 ; 20",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "(First Revision) 20 0 5.1 ; 20",
    },
    "IS 2585: 1968": {
        "title":    "BLACK SQUARE BOLTS AND NUTS (DIAMETER RANGE 6 TO 39 mm) AND BLACK SQUARE SCREWS (DIAMETER RANGE 6 TO 24 mm) For detailed information, refer to Specifi",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "(DIAMETER RANGE 6 TO 24 mm) (First Revision) For detailed information, refer to IS 2585 : 1968 Specification for black square bolts and nuts (diameter range 6 to 39 mm) and black square screws (diameter range 6 to 24 mm) (first revision). * Range of preferred lengths for bolts (bolts with lengths le",
    },
    "IS 2687: 1991": {
        "title":    "CAP NUTS",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Specification for CAP NUTS. Covers materials, dimensions, physical and mechanical requirements.",
    },
    "IS 2907: 1998": {
        "title":    "NON-FERROUS RIVETS ( TO 10 mm) 1. Scope \u2013 Requirements of copper, tinned copper, brass and aluminium rivets in the diameter range of 1mm to 10mm, inte",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements of copper, tinned copper, brass and aluminium rivets in the diameter range of 1mm to 10mm, intended for general engineering purposes.",
    },
    "IS 2998: 1982": {
        "title":    "COLD FORGED STEEL RIVETS FOR COLD CLOSING (1 TO 16 MM DIAMETER) For detailed information, refer to Specification for cold forged steel rivets for cold",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "(First Revision) For detailed information, refer to IS 2998 : 1982 Specification for cold forged steel rivets for cold closing (1 to 16 mm diameter) (first revision). Note 1\u2013 For detailed dimensions, refer to Tables 1 to 4 of the standard. Note 2\u2013 For general requirements for supply of rivets and th",
    },
    "IS 3063: 1994": {
        "title":    "FASTENERS-SINGLE COIL RECTANGULAR SECTION SPRING LOCK WASHERS 1. Scope \u2013 Requirements for single coil rectangular section spring lock washers suitable",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for single coil rectangular section spring lock washers suitable for use with bolt/ nut assemblies involving fasteners of property class 5.8 or less in the size range 2 to 100 mm.",
    },
    "IS 3121: 1981": {
        "title":    "RIGGING SCREWS AND STRETCHING SCREWS 1. Scope \u2013 Requirements regarding materials, components, dimensions, finishing and tests for rigging screws and s",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements regarding materials, components, dimensions, finishing and tests for rigging screws and stretching screws (double-ended and single- ended) of the following nominal size : a) Rigging screws \u2013 M12 to M90 b) Stretching screws \u2013 M6 to M52",
    },
    "IS 3468: 1991": {
        "title":    "PIPE NUTS",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Specification for PIPE NUTS. Covers materials, dimensions, physical and mechanical requirements.",
    },
    "IS 3757: 1985": {
        "title":    "HIGH STRENGTH STRUCTURAL BOLTS 1. Scope \u2013 Requirements for large series hexagon, high strength structural steel bolts in property classes and and in t",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for large series hexagon, high strength structural steel bolts in property classes 8.8 and 10.9 and in the size range M16 to M36 with short thread lengths suitable for use in both friction-type and bearing-type structural steel joints. Bolts to this standard when matched with the appropriate nuts have been designed to provide an assembly with a high level of assurance against failure by thread stripping on overtightening.",
    },
    "IS 4762: 1984": {
        "title":    "WORM DRIVE CLAMPS FOR GENERAL PURPOSE 1. Scope \u2013 Requirements for worm drive hose clamps for general purposes. 2. Size \u2013 12, 16, 20, 22, 25,28, 30, 35",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for worm drive hose clamps for general purposes.",
    },
    "IS 5369: 1975": {
        "title":    "GENERAL REQUIREMENTS FOR PLAIN WASHERS AND LOCK WASHERS 1. Scope \u2014 General requirements and permissible deviation for plain washers, lock washers and",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "General requirements and permissible deviation for plain washers, lock washers and similar parts.",
    },
    "IS 5372: 1975": {
        "title":    "TAPER WASHERS FOR CHANNELS (ISMC)",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "TAPER WASHERS FOR CHANNELS (ISMC)",
    },
    "IS 5373: 1969": {
        "title":    "SQUARE WASHERS FOR WOOD FASTENINGS 1. Scope \u2013 Requirements for square washers intended for use in wood fastenings with bolts in diameter range 10 to 5",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for square washers intended for use in wood fastenings with bolts in diameter range 10 to 52 mm.",
    },
    "IS 5374: 1975": {
        "title":    "TAPER WASHERS FOR I-BEAMS (ISMB) 1. Scope \u2013 Requirements for taper washers for use with Indian Standard Medium Weight Beams (ISMB) with bolts of 10 mm",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for taper washers for use with Indian Standard Medium Weight Beams (ISMB) with bolts of 10 mm to 39 mm diameter.",
    },
    "IS 5624: 1993": {
        "title":    "FOUNDATION BOLTS",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "FOUNDATION BOLTS 1. Scope \u2013 Requirements for J-type hook bolts and nuts, mushroom head seam bolts and nuts, and washers of aluminium for roofing sheets. 2. Requirements Material \u2013 Aluminium and aluminium alloys, as specified in the standard. Size Nominal Dia Thread Length, Size of Nut Preferred Lengths Min M6 6 25 M6 70, 80, 90, 100, 110, 120, 130, 140 and 150 M8 8 25 M8 70, 80, 90, 100, 110, 120,",
    },
    "IS 6113: 1970": {
        "title":    "ALUMINIUM FASTENERS FOR BUILDING PURPOSES 4. Designation \u2014 As an example, seam bolt size M8, length 20 mm and material HG 19 shall be designated as `S",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "4. Designation \u2014 As an example, seam bolt size M8, length 20 mm and material HG 19 shall be designated as `Seam Bolts M8 x 20, IS 6113\u2014 HG 19\u2019.",
    },
    "IS 6610: 1972": {
        "title":    "HEAVY WASHERS FOR STEEL STRUCTURES 2. Dimensions (in mm) 4. Designation \u2013 By nominal size and the number of this standard. Example \u2013 Washer 14 IS 6610",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Dimensions (in mm) 4. Designation \u2013 By nominal size and the number of this standard. Example \u2013 Washer 14 IS 6610.",
    },
    "IS 6623: 2004": {
        "title":    "HIGH STRENGTH STRUCTURAL NUTS 1. Scope \u2013 Requirements for large series hexagon, high strength structural steel nuts in property classes 8 and 10 and i",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for large series hexagon, high strength structural steel nuts in property classes 8 and 10 and in the size range M16 to M36 suitable for use in both friction-type and bearing-type structural steel connections. Nuts to this standard when matched with the appropriate bolts have been designed to provide an assembly with a high level of assurance against failure by thread stripping on overtightening.",
    },
    "IS 6639: 1972": {
        "title":    "HEXAGON BOLTS FOR STEEL STRUCTURES",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "HEXAGON BOLTS FOR STEEL STRUCTURES",
    },
    "IS 6649: 1985": {
        "title":    "HARDENED AND TEMPERED WASHERS FOR HIGH STRENGTH STRUCTURAL BOLTS AND NUTS",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "STRENGTH STRUCTURAL BOLTS AND NUTS",
    },
    "IS 6733: 1972": {
        "title":    "WALL AND ROOFING NAILS",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "WALL AND ROOFING NAILS For detailed information, refer to Specification for slotted raised countersunk head wood screws. 1. Scope \u2013 Requirements for slotted raised countersunk head wood screws. 2. Dimensions (in mm) No. Dia of Unthreaded shank Range of Screw Preferred Lengths Designation) Nom Max Min (see Note1) 0 8-12 1 8-12 2 8-12 3 8-12 4 12-25 5 12-30 6 12-40 7 12-40 8 12-75 9 15-75 10 15-75 1",
    },
    "IS 6736: 1972": {
        "title":    "SLOTTED RAISED COUNTERSUNK HEAD WOOD SCREWS",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "SLOTTED RAISED COUNTERSUNK HEAD WOOD SCREWS",
    },
    "IS 6739: 1972": {
        "title":    "SLOTTED ROUND HEAD WOOD SCREWS 1. Scope \u2013 Requirements for slotted round head wood screws. 2. Dimensions (in mm) No. Dia of Unthreaded Shank Range of",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Requirements for slotted round head wood screws.",
    },
    "IS 6760: 1972": {
        "title":    "SLOTTED COUNTERSUNK HEAD WOOD SCREWS",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "SLOTTED COUNTERSUNK HEAD WOOD SCREWS",
    },
    "IS 8033: 1976": {
        "title":    "WASHERS WITH SQUARE HOLE FOR WOOD FASTENINGS 4. Designation \u2013 As an example, a round washer with square hole of nominal size 14 mm shall be designated",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "4. Designation \u2013 As an example, a round washer with square hole of nominal size 14 mm shall be designated as `Washer With Square Hole 14 IS : 8033\u2019. * General requirements for plain washers and lock washers (first revision)",
    },
    "IS 8412: 1977": {
        "title":    "SLOTTED COUNTERSUNK HEAD BOLTS FOR STEEL STRUCTURES",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "SLOTTED COUNTERSUNK HEAD BOLTS FOR STEEL STRUCTURES",
    },
    "IS 8869: 1978": {
        "title":    "WASHERS FOR CORRUGATED SHEET ROOFING",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "WASHERS FOR CORRUGATED SHEET ROOFING",
    },
    "IS 10238: 2001": {
        "title":    "FASTENERS \u2013 THREADED STEEL FASTENERS \u2013 STEP BOLT FOR STEEL STRUCTURES For detailed information, refer to Specification for Fasteners \u2013 Threaded steel",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "1.1 Covers the requirements for step bolt used in steel structures including transmission towers to gain access to the top. 1.2 Each bolt shall be supplied with two hexagon nuts.",
    },
    "IS 12427: 2001": {
        "title":    "FASTENERS \u2013 THREADED STEEL FASTENERS \u2013 HEXAGON HEAD TRANSMISSION TOWER BOLTS 1. Scope \u2013 Covers the requirements for hot-dip galvanized hexagon head tr",
        "category": "Threaded Fasteners and Rivets",
        "section":  19,
        "scope":    "Covers the requirements for hot-dip galvanized hexagon head transmission tower bolts in the size range M12 to 24 for use in the construction of transmission towers, sub-stations and similar steel structures 1.1 The bolts covered in this standard are not suitable for applications requiring improved low temperatures characteristics.",
    },
    "IS 278: 1978": {
        "title":    "GALVANIZED STEEL BARBED WIRE FOR FENCING 1. Scope \u2013 Requirements for two types of galvanized steel barbed wire with two strands of wire. 2. Types \u2013 Ty",
        "category": "Wire Ropes and Wire Products",
        "section":  20,
        "scope":    "Requirements for two types of galvanized steel barbed wire with two strands of wire.",
    },
    "IS 2140: 1978": {
        "title":    "STRANDED GALVANIZED STEEL WIRE FOR FENCING 1. Scope \u2013 Requirements for galvanized strand fencing wire of 3-ply and 7-ply construction. For detailed in",
        "category": "Wire Ropes and Wire Products",
        "section":  20,
        "scope":    "Requirements for galvanized strand fencing wire of 3-ply and 7-ply construction. For detailed information, refer to IS 2140 : 1978 Specifications for stranded galvanized steel wire for fencing (first revision).",
    },
    "IS 2365: 1977": {
        "title":    "STEEL WIRE SUSPENSION ROPES FOR LIFTS, ELEVATORS AND HOISTS 1. Scope \u2013 Requirements of steel wire ropes for use with lifts, elevators and hoists havin",
        "category": "Wire Ropes and Wire Products",
        "section":  20,
        "scope":    "Requirements of steel wire ropes for use with lifts, elevators and hoists having cars or platforms carrying passengers or goods and working in guides. Does not apply to ropes used for winding purposes in mines. Following rope constructions and size ranges are covered: Nominal Approximate Range of Minimum Breaking Load Corresponsing Diameter Mass Range to Tensile Designation of Wires of mm kg/100m 1230 1420 1570 kN kN kN 6 12.5 t o 13.7 13.6 t o 1",
    },
    "IS 2721: 1979": {
        "title":    "GALVANIZED STEEL CHAIN LINK FENCE FABRIC 1. Scope \u2013 Requirements for galvanized steel chain fence fabric intended for various purposes. This standard",
        "category": "Wire Ropes and Wire Products",
        "section":  20,
        "scope":    "Requirements for galvanized steel chain fence fabric intended for various purposes. This standard does not cover the requirements pertaining to straining posts, struts, base plates and other fittings. *Mild steel wire for general engineering purposes (third revision)",
    },
    "IS 2553 (PART 1): 1990": {
        "title":    "SAFETY GLASS PART 1 GENERAL PURPOSE 1. Scope \u2013 Requirements and the methods of sampling and test for safety glass meant for general purposes such as f",
        "category": "Glass",
        "section":  21,
        "scope":    "Requirements and the methods of sampling and test for safety glass meant for general purposes such as for use in glazing windows, doors of buildings and railway coaches.",
    },
    "IS 2835: 1987": {
        "title":    "FLAT TRANSPARENT SHEET GLASS 1. Scope \u2013 Requirements and methods of sampling and test for flat transparent sheet glass for use in the manufacture of p",
        "category": "Glass",
        "section":  21,
        "scope":    "Requirements and methods of sampling and test for flat transparent sheet glass for use in the manufacture of photographic plates, projection slides, silvered glass mirrors, toughened or laminated safety glasses and for glazing and framing purposes.",
    },
    "IS 3438: 1994": {
        "title":    "SILVERED GLASS MIRRORS FOR GENERAL PURPOSES 1. Scope \u2013 Requirements and methods of sampling and test for silvered glass mirrors used for general purpo",
        "category": "Glass",
        "section":  21,
        "scope":    "Requirements and methods of sampling and test for silvered glass mirrors used for general purposes.",
    },
    "IS 5437: 1994": {
        "title":    "FIGURED, ROLLED AND WIRED GLASS 1. Scope \u2013 Requirements and methods of sampling and test for figured, rolled and wired glass. 2. General Requirements",
        "category": "Glass",
        "section":  21,
        "scope":    "Requirements and methods of sampling and test for figured, rolled and wired glass.",
    },
    "IS 110: 1983": {
        "title":    "READY MIXED PAINT, BRUSHING, GREY FILLER, FOR ENAMELS FOR USE OVER PRIMERS 1. Scope \u2013 Requirements, and the methods of sampling and test for ready mix",
        "category": "Fillers, Stoppers and Putties",
        "section":  22,
        "scope":    "Requirements, and the methods of sampling and test for ready mixed paint, brushing, grey filler, for enamels, for use over primers. The material is used as a filler over the primer in the painting system normally followed by enamels.",
    },
    "IS 419: 1967": {
        "title":    "PUTTY, FOR USE ON WINDOW FRAMES 1. Scope \u2013 Requirements, and the methods of sampling and test for putty for use in fixing glass panes on wood and meta",
        "category": "Fillers, Stoppers and Putties",
        "section":  22,
        "scope":    "Requirements, and the methods of sampling and test for putty for use in fixing glass panes on wood and metal frames and for filling splits, cracks and holes in wood or metal.",
    },
    "IS 423: 1961": {
        "title":    "PLASTIC WOOD FOR JOINERS FILLER (Revised) 1. Scope \u2013 Requirements, and methods of test for material commercially known as plastic wood, for joiners fi",
        "category": "Fillers, Stoppers and Putties",
        "section":  22,
        "scope":    "Requirements, and methods of test for material commercially known as plastic wood, for joiners fillers. The material is used for filling holes, cracks and other irregularities in wood to produce a smooth surface capable of taking suitable stain to match timber.",
    },
    "IS 3709: 1966": {
        "title":    "MASTIC CEMENT FOR BEDDING OF METAL WINDOW 1. Scope \u2013 Requirements and methods of sampling and test for mastic cement for bedding of metal windows. The",
        "category": "Fillers, Stoppers and Putties",
        "section":  22,
        "scope":    "Requirements and methods of sampling and test for mastic cement for bedding of metal windows. The material is intended for application by hand or with a putty knife. It is used for bedding, one metal window into another, metal windows into wooden frames, or metal frames into masonry or concrete. It is expected to be suitable for taking paint without lifting, bleeding or cracking.",
    },
    "IS 3677: 1985": {
        "title":    "UNBONDED ROCK AND SLAG WOOL FOR THERMAL INSULATION 1. Scope \u2013 Requirements and the methods of sampling and test for unbonded rock and slag wool for th",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and the methods of sampling and test for unbonded rock and slag wool for thermal insulation.",
    },
    "IS 4671: 1984": {
        "title":    "EXPANDED POLYSTYRENE FOR THERMAL INSULATION PURPOSES 1. Scope \u2013 Requirements and the methods of sampling and test for expanded polystyrene in the form",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and the methods of sampling and test for expanded polystyrene in the form of rough shapes, finished boards and blocks, and pipe sections / segments for thermal insulation primarily for use in refrigeration and building applications in the temperature range - 150\u00ba to 80\u00baC.",
    },
    "IS 6598: 1972": {
        "title":    "CELLULAR CONCRETE FOR THERMAL INSULATION 1. Scope \u2013 Requirements and the methods of sampling and test for cellular concrete for thermal linsulation. N",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and the methods of sampling and test for cellular concrete for thermal linsulation.",
    },
    "IS 7509: 1993": {
        "title":    "THERMAL INSULATING CEMENT 1. Scope \u2013 Requirements and the methods of sampling and test for thermal linsulating cements for use at temperatures up to 9",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and the methods of sampling and test for thermal linsulating cements for use at temperatures up to 950\u00baC.",
    },
    "IS 8154: 1993": {
        "title":    "PREORMED CALCIUM SILICATE INSULATION (FOR TEMPERATURES UPTO 650 0C) 1. Scope \u2013 Requirements and the methods of sampling and test for performed calcium",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and the methods of sampling and test for performed calcium silicate insulation intended for use on surface which reach temperatures upto 6500C.",
    },
    "IS 8183: 1993": {
        "title":    "BONDED MINERAL WOOL 1. Scope \u2013 Requirements and the methods of sampling and test for bonded mineral wool for thermal insulation. 2. Requirements Descr",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and the methods of sampling and test for bonded mineral wool for thermal insulation.",
    },
    "IS 9428: 1993": {
        "title":    "PREFORMED CALCIUM SILICATE INSULATION (FOR TEMPERATURES UP TO 950\u00baC) 1. Scope \u2013 Requirements and the methods of sampling and test for preformed calciu",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and the methods of sampling and test for preformed calcium silicate insulation intended for use with surfaces with reach temperatures up to 9500C.",
    },
    "IS 9742: 1993": {
        "title":    "SPRAYED MINERAL WOOL THERMAL INSULATION 1. Scope \u2013 Requirements and the methods of sampling and test for sprayed mineral wool thermal insulation. 2. M",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and the methods of sampling and test for sprayed mineral wool thermal insulation.",
    },
    "IS 9743: 1990": {
        "title":    "THERMAL INSULATION FINISHING CEMENTS 1. Scope Requirements for thermal insulation finishing cements, prepared by mixing with water for application to",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "1.1 Requirements for thermal insulation finishing cements, prepared by mixing with water for application to insulating materials after they have been applied at site to the plant of piping systems.",
    },
    "IS 9842: 1994": {
        "title":    "PREFORMED FIBROUS PIPE INSULATION 1. Scope \u2013 Requirements and methods of sampling and test for preformed fibrous pipe sections for thermal imsulation.",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and methods of sampling and test for preformed fibrous pipe sections for thermal imsulation.",
    },
    "IS 11128: 1984": {
        "title":    "SPRAY \u2013 APPLIED HYDRATED CALCIUM SILICATE THERMAL INSULATION 1. Scope \u2013 Requirements and methods of sampling and test for spray-applied, hydrated calc",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and methods of sampling and test for spray-applied, hydrated calcium silicate thermal insulation.",
    },
    "IS 11307: 1985": {
        "title":    "CELLULAR GLASS BLOCK AND PIPE THERMAL INSULATION 1. Scope \u2013 Requirements and methods of sampling and test for cellular glass block and pipe thermal in",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and methods of sampling and test for cellular glass block and pipe thermal insulation intended for use on surfaces operating at temperatures between \u2013 2000C and 4250C.",
    },
    "IS 11308: 1985": {
        "title":    "HYDRAULIC SETTING THERMAL INSULATING CASTABLES FOR TEMPERATURE UPTO 1250\u00baC 1. Scope \u2013 Requirements and methods of sampling and test for hydraulic sett",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and methods of sampling and test for hydraulic setting thermal insulating castables for use as either hot face or cold face backing of refractory linings, at temperatures up to 12500C.",
    },
    "IS 12436: 1988": {
        "title":    "PREFORMED RIGID POLYURETHANE (PUR) AND POLYISOCYANURATE (PIR) FOAMS FOR THERMAL INSULATION 1. Scope \u2013 Requirements, and the methods of sampling and te",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements, and the methods of sampling and test for preformed rigid polyurethane (PUR) and polyisocyanurate (PIR) foam for thermal insulation in the form of boards, cut and moulded slabs, cut and moulded pipe sections, cut and moulded radiused and bevelled lags, panels with adhesive integrally laminated facings, panels with adhesive applied facings, and cut and moulded special shapes.",
    },
    "IS 13204: 1991": {
        "title":    "RIGID PHENOLIC FOAM FOR THERMAL INSULATION 1. Scope \u2013 Requirements and methods of sampling and test for rigid for phenolic foam for thermal insulation",
        "category": "Thermal Insulation Materials",
        "section":  23,
        "scope":    "Requirements and methods of sampling and test for rigid for phenolic foam for thermal insulation purposes. It applies to slab (blocks, boards and profiled sheets) and profiled sections (pipe sections and radiused or bevelled lags) cut from pipes. The nominal temperature range for which the insulation material is suitable is \u2013180 to + 130\u00baC without any facing. The material is normally supplied with craft paper facing on both sides. This standard i",
    },
    "IS 2036: 1995": {
        "title":    "PHENOLIC LAMINATED SHEETS 1. Scope Requirements and the methods of sampling and test for phenolic resin bonded laminated sheets of one class in which",
        "category": "Plastics",
        "section":  24,
        "scope":    "1.1 Requirements and the methods of sampling and test for phenolic resin bonded laminated sheets of one class in which the mechanical properties in directions A and B are of the same order, with asbestos, woven cotton fabric and cellulose paper reinforcements and covers seventeen types. 1.2 Covers only sheets of a nominal thickness from 0.4 to 50 mm for cellulose paper based types and nominal thickness from 0.4 mm to 100 mm for woven cotton fabri",
    },
    "IS 2046: 1995": {
        "title":    "DECORATIVE THERMOSETTING SYNTHETIC RESIN BONDED LAMINATED SHEETS 1. Scope \u2013 Requirements and the methods of sampling and test for decorative laminated",
        "category": "Plastics",
        "section":  24,
        "scope":    "Requirements and the methods of sampling and test for decorative laminated sheets (HPL) classified according to their performance and main recommended fields of application and provides also for materials of special characteristics, for example, post formability or defined reaction to fire. They are intended for interior applications.",
    },
    "IS 2508: 1984": {
        "title":    "LOW DENSITY POLYTHYLENE FILMS 1. Scope Requirements and methods of sampling and test for natural and black colour (carbon black pigment) low density p",
        "category": "Plastics",
        "section":  24,
        "scope":    "1.1 Requirements and methods of sampling and test for natural and black colour (carbon black pigment) low density polyethylene films intended for packaging, canal lining, agricultural operations and post harvest uses, construction work and allied purposes. 1.2 This standard cover flexible, unsupported flat or tubular films 12.5 to 250 mm in thickness and width 175 to 7 500 mm (350 to 15 000 mm slit open width in the case of tubular films), made f",
    },
    "IS 6307: 1985": {
        "title":    "RIGID PVC SHEETS 1. Scope \u2013 Requirements and methods of sampling and test for rigid PVC sheets of to mm in thickness, manufactured by calendering, ext",
        "category": "Plastics",
        "section":  24,
        "scope":    "Requirements and methods of sampling and test for rigid PVC sheets of 0.10 to 12.5 mm in thickness, manufactured by calendering, extrusion or calendering followed by lamination.",
    },
    "IS 10889: 1984": {
        "title":    "HIGH DENSITY POLYTHYLENE FILM 1. Scope \u2013 Requirements and the methods of sampling and test for natural and black colour (carbon black pigment) high de",
        "category": "Plastics",
        "section":  24,
        "scope":    "Requirements and the methods of sampling and test for natural and black colour (carbon black pigment) high density polyethylene films. Coloured films other than black shall be as a greed to between the purchaser and the supplier.",
    },
    "IS 12830: 1989": {
        "title":    "RUBBER BASED ADHESIVE FOR FIXING PVC TILES TO CEMENT 1. Scope \u2013 Requirements and the methods of sampling and test for rubber based adhesives used for",
        "category": "Plastics",
        "section":  24,
        "scope":    "Requirements and the methods of sampling and test for rubber based adhesives used for bonding PVC tiles to cement, floors and walls of buildings.",
    },
    "IS 12994: 1990": {
        "title":    "EPOXY ADHESIVES, ROOM TEMPERATURE CURING, GENERAL PURPOSE 1. Scope \u2013 Requirements and methods of sampling and test for liquid and paste type epoxy adh",
        "category": "Plastics",
        "section":  24,
        "scope":    "Requirements and methods of sampling and test for liquid and paste type epoxy adhesives for performance : (a) up to 50\u00baC, and (b) up to 100\u00baC.",
    },
    "IS 14182: 1994": {
        "title":    "SOLVENT CEMENT FOR USE WITH UNPLASTICIZED POLYVINYL CHLORIDE PLASTIC PIPE AND FITTINGS 1. Scope \u2013 Requirements and methods of sampling and test for so",
        "category": "Plastics",
        "section":  24,
        "scope":    "Requirements and methods of sampling and test for solvent cements to be used in joining unplasticized polyvinyl chloride pipe and fittings intended for use in carrying potable water. The pipes may be pressure or non-pressure type.",
    },
    "IS 14443: 1997": {
        "title":    "POLYCARBONATE SHEETS 1. Scope Requirements and methods of sampling and tests for polycarbonate sheets of solid section as well as multi-wall variety a",
        "category": "Plastics",
        "section":  24,
        "scope":    "1.1 Requirements and methods of sampling and tests for polycarbonate sheets of solid section as well as multi-wall variety and also thinner gauge sheets (films), multi-layer composite laminates of polycarbonate compact sheets and composites of polycarbonate compact sheets and glass sheets. Sheets containing glass fibre or any other reinforcement are, however, not covered by this standard. 1.2 This standard establishes a system for designating vac",
    },
    "IS 14643: 1999": {
        "title":    "UNSINTERED POLYTETRAFLUOROETHYLENE (PTFE) TAPE FOR THREADS SEALING APPLICATIONS 1. Scope \u2013 Requirements, methods of sampling and tests for unsintered",
        "category": "Plastics",
        "section":  24,
        "scope":    "Requirements, methods of sampling and tests for unsintered polytetrafluoroethylene (PTFE) tapes for use as a thread sealing material and in similar applications. This tape is suitable for applications under ambient conditions with all common fluids and gases up to 80 bar gauge for pipes. This tape is suitable for applications in the range \u2014200C to 2000C for pipe sizes up to 50 mm.",
    },
    "IS 14753: 1999": {
        "title":    "POLYMETHYL METHACRYLATE (PMMA) (ACRYLIC) SHEETS 1. Scope \u2013 Requirements, methods of sampling and tests for polymethyl methacrylate (acrylic) sheets. 2",
        "category": "Plastics",
        "section":  24,
        "scope":    "Requirements, methods of sampling and tests for polymethyl methacrylate (acrylic) sheets.",
    },
    "IS 694: 1990": {
        "title":    "PVC INSULATED CABLE FOR WORKING VOLTAGES UP TO AND INCLUDING 1100 VOLTS 1. Scope Requirements and tests for the following types of unarmoured PVC insu",
        "category": "Conductors and Cables",
        "section":  25,
        "scope":    "1.1 Requirements and tests for the following types of unarmoured PVC insulated cables with copper or aluminium conductors and flexible cords with copper conductors for electric power and lighting (including cables for outdoor use and cables for low temperature conditions) for voltages up to and including 1100 Volts. Note 1 For out door use, the cables shall meet the requirements of additional ageing test. [see 15.4 of the standard] Note 2 The cab",
    },
    "IS 1554 (PART 1): 1988": {
        "title":    "PVC INSULATED (HEAVY DUTY) ELECTRIC CABLES PART 1 \u2013 FOR WORKING VOLTAGES UPTO AND INCLUDING 1 100 VOLTS \u2020PVC insulation and sheath of electric cables",
        "category": "Conductors and Cables",
        "section":  25,
        "scope":    "PART 1 \u2013 FOR WORKING VOLTAGES UPTO AND INCLUDING 1 100 VOLTS (Third Revision) \u2020PVC insulation and sheath of electric cables (first revision).",
    },
    "IS 7098 (PART 1): 1988": {
        "title":    "CROSSLINKED POLYETHYLENE INSULATED THERMOPLASTIC SHEATHED CABLES PART-1 FOR WORKING VOLTAGES UP TO AND INCLUDING 1 100 VOLTS 1. Scope Requirements for",
        "category": "Conductors and Cables",
        "section":  25,
        "scope":    "1.1 Requirements for both armoured and unarmoured single, twin, three, four and multi-core cross\u2013linked polyethylene (XLPE) insulated and PVC sheathed cables for electric supply and control purpose. 1.2 The cables covered in this standard are suitable for use on ac single phase or three phase (earthed or unearthed) systems for rated voltages up to and including 1 100 volts. These cables may be used on dc systems for rated voltage up to and includ",
    },
    "IS 9968 (PART 1): 1998": {
        "title":    "ELASTOMER INSULATED CABLES, PART1 FOR WORKING VOLTAGES UPTO AND INCLUDING 1 100 VOLTS 1. Scope \u2013 Requirements of elastomeric insulated cables for fixe",
        "category": "Conductors and Cables",
        "section":  25,
        "scope":    "Requirements of elastomeric insulated cables for fixed wiring, flexible cables and flexible cords for electric power and lighting for operation at voltages up to and including 1100 volts. 1.1 The following types of cables and cords are covered in this standard. 1.1.1 Cables for fixed wiring a) Braided and compounded/varnished, b) Elastomer sheathed (normal duty), and c) Elastomer sheathed (normal duty) with earth continuity conductor. 1.1.2 Flexi",
    },
    "IS 1293: 1988": {
        "title":    "PLUGS AND SOCKET-OUTLETS \u2013 RATED VOLTAGE UPTO AND INCLUDING 250 VOLTS AND RATED CURRENT UPTO AND INCLUDING 16 AMPERES 1. Scope \u2013 Requirements and test",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "Requirements and tests for three-pin two- pole and earthing plugs and socket-outlets (shuttered and non-shuttered) including multi-socket-outlet (shuttered and non-shuttered) suitable for ac circuits with a rated voltage above 50 V but not exceeding 250 volts and a rated current of 6 A or 16 A. Note 1\u2013 2 pin plugs and socket outlets are considered non- standard. Note 2\u2013 Fused plugs are not covered under the scope of this standard.",
    },
    "IS 2086: 1993": {
        "title":    "CARRIERS AND BASES USED IN REWIRABLE TYPE ELECTRIC FUSES FOR VOLTAGES UP TO 650 VOLTS 1. Scope \u2013 Performance requirements and tests as well as dimensi",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "Performance requirements and tests as well as dimensions of carriers and bases used in rewirable type electric fuses having a rated current up to and including 100 A meant for alternating current systems of voltages not exceeding 650v between lines.The specification does not cover fuse-wire used in rewirable type fuses.",
    },
    "IS 2412: 1975": {
        "title":    "LINK CLIPS FOR ELECTRICAL WIRING 1. Scope \u2013 Requirements and tests for link clips (both joint link clips and link clips with separate linking eyes) fo",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "Requirements and tests for link clips (both joint link clips and link clips with separate linking eyes) for general wiring purpose.",
    },
    "IS 3419: 1989": {
        "title":    "FITTINGS FOR RIGID NON-METALLIC CONDUITS 1. Scope \u2013 Requirements and methods of test for rigid conduit fittings manufactured from insulating materials",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "Requirements and methods of test for rigid conduit fittings manufactured from insulating materials for use with circular, rigid, non-flame propagating and non-threadable plain conduits of insulating materials. This standard covers conduit fittings suitable for temperature between \u20135\u00baC and + 60\u00baC. Only plain type fittings are covered in this standard. The fittings covered by this standard are\u2013couplers, bends, elbows,tees, inspection sleeves, and b",
    },
    "IS 3480: 1966": {
        "title":    "FLEXIBLE STEEL CONDUITS FOR ELECTRICAL WIRING TABLE 1 REQUIREMENTS FOR FLEXIBLE STEEL CONDUCTS NOMINAL INTERNAL TOLERANCE EXTERNAL TURNS",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "TABLE 1 REQUIREMENTS FOR FLEXIBLE STEEL CONDUCTS NOMINAL INTERNAL TOLERANCE EXTERNAL TURNS",
    },
    "IS 3837: 1976": {
        "title":    "ACCESSORIES FOR RIGID STEEL CONDUITS FOR ELECTRICAL WIRING Note 1 \u2013 Tolerance shall be \u00b15 percent on nominal dimensions. Note 2 \u2013 The material shall b",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "(First Revision) Note 1 \u2013 Tolerance shall be \u00b15 percent on nominal dimensions. Note 2 \u2013 The material shall be mild steel for clips, saddles, plugs and lock nuts, mild steel forgings for pipe hooks and crampets, and shall be moulded insulating for bushes.",
    },
    "IS 3854: 1997": {
        "title":    "SWITCHES FOR DOMESTIC AND SIMILAR PURPOSES",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "SWITCHES FOR DOMESTIC AND SIMILAR PURPOSES",
    },
    "IS 4160: 1967": {
        "title":    "INTERLOCKING SWITCH SOCKET OUTLET",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "INTERLOCKING SWITCH SOCKET OUTLET",
    },
    "IS 4615: 1968": {
        "title":    "SWITCH SOCKET-OUTLETS (NON-INTERLOCKING TYPE)",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "SWITCH SOCKET-OUTLETS (NON-INTERLOCKING TYPE)",
    },
    "IS 4649: 1968": {
        "title":    "ADAPTORS FOR FLEXIBLE STEEL CONDUITS For detailed information, refer to Specification for adaptors for flexible steel conduits",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "ADAPTORS FOR FLEXIBLE STEEL CONDUITS For detailed information, refer to Specification for adaptors for flexible steel conduits",
    },
    "IS 6538: 1971": {
        "title":    "THREE-PIN PLUGS MADE OF RESILIENTMATERIAL",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "THREE-PIN PLUGS MADE OF RESILIENTMATERIAL",
    },
    "IS 8828: 1996": {
        "title":    "/IEC 898 (1995) CIRCUIT BREAKERS FOR OVER CURRENT PROTECTION FOR HOUSEHOLD AND SIMILAR INSTALLATIONS 1. Scope \u2013 Applies to A.C. air-break circuit-brea",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "Applies to A.C. air-break circuit-breakers for operation at 50 Hz or 60 Hz, having a rated voltage not exceeding 440 V (between phases), a rated current not exceeding 125 A and a rated short-circuit capacity not exceeding 25 000 A. 1.2 These circuit-breakers are intended for the protection against",
    },
    "IS 9537 (PART 1): 1980": {
        "title":    "CONDUITS FOR ELECTRICAL INSTALLATIONS PART 1 - GENERAL REQUIREMENTS",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "for electrical installations \u2013 Part I General requirements. Rigid steel conduits.",
    },
    "IS 9537 (PART 3): 1983": {
        "title":    "CONDUITS FOR ELECTRICAL INSTALLATIONS PART 3 \u2013 RIGID PLAIN CONDUITS OF INSULATING MATERIALS",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "PART 3 \u2013 RIGID PLAIN CONDUITS OF INSULATING MATERIALS",
    },
    "IS 9537 (PART 4): 1983": {
        "title":    "CONDUITS FOR ELECTRICAL INSTALLATIONS PART 4 \u2013 PLIABLE SELF RECOVERING CONDUITS OF INSULATING MATERIALS Note \u2013 For details requirements refer to the s",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "PART 4 \u2013 PLIABLE SELF RECOVERING CONDUITS OF INSULATING MATERIALS Note \u2013 For details requirements refer to the standard For detailed information, refer to IS 9537 (Part 4) : 1983 Specification for conduits for electrical installations :Part 4 Pliable self- recovering conduits of insulating materials",
    },
    "IS 9537 (PART 5): 2000": {
        "title":    "CONDUITS FOR ELECTRICAL INSTALLATIONS PART 5 \u2013 PLIABLE CONDUITS OF INSULATING MATERIAL 1. Scope \u2013 This clause of Part 1 of the Standard is applicable",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "This clause of Part 1 of the Standard is applicable except as follows: Addition: This Indian Standard (Part 5) specifies requirements for pliable non-flame propagating plain and corrugated conduits of insulating material. It does not include self- recovering or flexible conduits. This standard also applies to corrugated conduits with a smooth exterior surface.",
    },
    "IS 14772: 2000": {
        "title":    "GENERAL REQUIREMENTS FOR ENCLOSURES FOR ACCESSORIES FOR HOUSEHOLD AND SIMILAR FIXED ELECTRICAL INSTALLATIONS \u2013 SPECIFICAION 1. Scope \u2013 This standard a",
        "category": "Wiring Accessories",
        "section":  26,
        "scope":    "This standard applies to enclosures or parts of enclosures or parts for accessories with a rated voltage not exceeding 440 V intended for household or similar fixed electrical installatinos, either indoors or outdoors. This standard may be used as a guide for enclosures having a rated vltage up to 100 V. Enclosures complying with this standard are suitable for use, after installation, an ambient temperatures not normally exceeding 35oC, but occas",
    },
    "IS 875 (PART 1): 1987": {
        "title":    "CODE OF PRACTICE FOR DESIGN LOADS (OTHERTHAN EARTHQUAKE)FORBUILDINGSANDSTRUCTURES, PART-1 DEAD LOADS \u2013 UNIT WEIGHTS OF BUILDING MATERIALS AND STORED M",
        "category": "General",
        "section":  27,
        "scope":    "Covers unit weight mass of materials, and parts or components in a building that apply to the determination of dead loads in the design of buildings. The unit weight mass of materials that are likely to be stored in a building are also specified in the standard for the purpose of load calculations along with angles internal friction as appropriate.",
    },
}
# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL ID MAP
# ══════════════════════════════════════════════════════════════════════════════

def _make_canonical(s):
    s = str(s).upper().strip()
    s = re.sub(r"\s+", "", s)
    return s

CANONICAL_MAP = {_make_canonical(k): k for k in STANDARDS_DB}

def normalise_id(raw):
    c = _make_canonical(raw)
    if c in CANONICAL_MAP:
        return CANONICAL_MAP[c]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION — 80+ rules covering all 27 sections
# ══════════════════════════════════════════════════════════════════════════════

EXPANSION_RULES = [
    # ── Section 1: OPC grades ────────────────────────────────────────────────
    (r"\b33\s*grade\b",                        "ordinary portland cement 33 grade IS 269 OPC"),
    (r"\b43\s*grade\b",                        "ordinary portland cement 43 grade IS 8112 OPC"),
    (r"\b53\s*grade\b",                        "ordinary portland cement 53 grade IS 12269 OPC"),
    (r"\bopc\s*33\b",                          "ordinary portland cement 33 grade IS 269"),
    (r"\bopc\s*43\b",                          "ordinary portland cement 43 grade IS 8112"),
    (r"\bopc\s*53\b",                          "ordinary portland cement 53 grade IS 12269"),
    (r"\b(?:opc|ordinary\s+portland\s+cement)\b","ordinary portland cement OPC IS 269"),
    (r"\b(?:psc|portland\s+slag\s+cement)\b",  "portland slag cement PSC blast furnace IS 455"),
    (r"\bsrpc\b",                              "sulphate resisting portland cement IS 12330"),
    (r"\brhpc\b",                              "rapid hardening portland cement IS 8041"),
    (r"\bhac\b",                               "high alumina cement structural IS 6452"),
    (r"\bssc\b",                               "supersulphated cement marine IS 6909"),
    (r"fly\s*ash.*(?:cement|ppc|pozzolana)",   "portland pozzolana cement fly ash IS 1489 Part 1"),
    (r"calcined\s*clay.*(?:cement|ppc)",       "portland pozzolana cement calcined clay IS 1489 Part 2"),
    (r"white\s*(?:portland\s*)?cement",        "white portland cement architectural decorative IS 8042"),
    (r"masonry\s*cement",                      "masonry cement general purpose mortar not structural IS 3466"),
    (r"hydrophobic\s*cement",                  "hydrophobic portland cement storage IS 8043"),
    (r"rapid\s*hard\w+\s*cement",              "rapid hardening portland cement IS 8041"),
    (r"high\s*alumina\s*cement",               "high alumina cement structural cold IS 6452"),
    (r"supersulphat\w+",                       "supersulphated cement marine aggressive IS 6909"),
    (r"sulphate\s*resist\w+\s*cement",         "sulphate resisting portland cement IS 12330"),
    (r"portland\s*slag\s*cement",              "portland slag cement blast furnace IS 455"),
    (r"(?:marine|coastal|sea\s*water)\s*(?:cement|work|structure)", "supersulphated cement marine IS 6909"),
    (r"sulphate\s*(?:soil|bearing|attack)",    "sulphate resisting portland cement IS 12330"),
    # ── Section 1: Aggregates ───────────────────────────────────────────────
    (r"(?:coarse|fine)\s*aggregate.*(?:concrete|structural)", "coarse fine aggregates natural sources IS 383"),
    (r"natural\s*(?:source\s*)?aggregate",     "coarse fine aggregates natural sources concrete IS 383"),
    (r"sand.*(?:mortar|masonry|brickwork)",    "sand for masonry mortars IS 2116"),
    (r"sand.*(?:plaster|plastering)",          "sand for plaster IS 1542"),
    (r"lightweight\s*aggregate",               "artificial lightweight aggregate IS 9142"),
    # ── Section 1: Concrete blocks ──────────────────────────────────────────
    (r"aac\s*block|autoclaved\s*aerated\s*concrete\s*block", "AAC blocks IS 2185 Part 3 autoclaved aerated thermal"),
    (r"lightweight.*(?:masonry|concrete)\s*block", "hollow solid lightweight concrete blocks IS 2185 Part 2"),
    (r"hollow.*solid.*lightweight.*block",     "IS 2185 Part 2 lightweight concrete blocks"),
    (r"hollow.*(?:concrete\s*)?block",         "hollow concrete blocks IS 2185 Part 1"),
    # ── Section 1: Pipes ────────────────────────────────────────────────────
    (r"precast\s*concrete\s*pipe",             "precast concrete pipes IS 458 reinforced unreinforced"),
    (r"(?:water\s*main|sewer|culvert|irrigation).*pipe", "precast concrete pipes IS 458"),
    (r"prestressed\s*concrete\s*pipe",         "prestressed concrete pipes IS 784"),
    (r"asbestos\s*cement.*pressure.*pipe",     "asbestos cement pressure pipes IS 1592"),
    (r"corrugated.*asbestos\s*cement",         "corrugated asbestos cement sheets IS 459 roofing cladding"),
    # ── Section 2: Limes ────────────────────────────────────────────────────
    (r"building\s*lime|hydraulic\s*lime|fat\s*lime|slaked\s*lime|quick\s*lime", "building limes IS 712 hydraulic fat"),
    # ── Section 3: Stones ───────────────────────────────────────────────────
    (r"natural\s*(?:building\s*)?stone|(?:granite|marble|sandstone|quartzite|slate|laterite)\s*(?:stone|masonry)?", "natural building stone masonry IS 1127"),
    (r"structural\s*granite",                  "structural granite IS 3316"),
    (r"stone\s*(?:lintel|sill)",               "stone lintels IS 9394"),
    # ── Section 4: Clay / Bricks ────────────────────────────────────────────
    (r"(?:common\s*)?burnt\s*clay\s*brick|common\s*brick", "common burnt clay bricks IS 1077"),
    (r"perforated\s*(?:clay\s*)?brick",        "burnt clay perforated bricks IS 2222"),
    (r"hollow\s*clay\s*(?:brick|block|tile)",  "burnt clay hollow bricks IS 3952"),
    (r"paving\s*brick|clay\s*paving",          "burnt clay paving bricks IS 3583"),
    (r"fly\s*ash\s*(?:brick|lime\s*brick)",    "fly ash lime bricks IS 12894"),
    (r"(?:clay\s*)?(?:ridge|ceiling)\s*tile",  "clay ridge ceiling tiles IS 1464"),
    (r"(?:clay\s*)?flooring\s*tile",           "clay flooring tiles IS 1478"),
    (r"(?:clay|burnt\s*clay)\s*(?:flat|terracing)\s*tile", "burnt clay terracing tiles IS 2690"),
    # ── Section 5: Gypsum ───────────────────────────────────────────────────
    (r"gypsum.*(?:plaster|board|tile|block)",  "gypsum plaster IS 2547 gypsum board"),
    (r"plaster\s*board",                        "gypsum plaster board"),
    # ── Section 6: Timber ───────────────────────────────────────────────────
    (r"(?:commercial\s*)?timber\s*(?:classification|grading|species)", "commercial timber IS 399"),
    (r"plywood",                                "plywood IS 303 IS 710"),
    (r"structural\s*timber",                    "structural timber IS 883"),
    # ── Section 7: Bitumen ──────────────────────────────────────────────────
    (r"paving\s*bitumen|road\s*bitumen",        "paving bitumen IS 73"),
    (r"bitumen\s*emulsion",                     "bitumen emulsion IS 8887"),
    (r"cutback\s*bitumen",                      "cutback bitumen IS 217"),
    (r"coal\s*tar|road\s*tar",                  "coal tar road tar IS 216"),
    (r"bituminous\s*(?:paint|coating)",         "bituminous paint IS 9862"),
    # ── Section 8: Flooring / Paint ─────────────────────────────────────────
    (r"cement\s*concrete\s*(?:floor\s*)?tile",  "cement concrete flooring tiles IS 1237"),
    (r"(?:ceramic|vitrified|glazed)\s*(?:floor|wall)\s*tile", "ceramic floor wall tiles IS 13630 IS 15622"),
    (r"(?:marble|granite)\s*(?:tile|slab)\s*(?:floor|polish)", "marble granite floor IS 1130"),
    (r"(?:internal|exterior)\s*paint|distemper|primer|enamel", "paint distemper IS 427 IS 428"),
    (r"(?:cement|synthetic)\s*paint",           "cement paint IS 5410 IS 427"),
    (r"linoleum",                               "linoleum IS 653"),
    # ── Section 9: Waterproofing ────────────────────────────────────────────
    (r"waterproof\w*\s*(?:membrane|compound|treatment)", "waterproofing IS 1322 IS 1346 membrane"),
    (r"damp\s*proof\w*|damp\s*proof\s*course",  "damp proofing course IS 3067"),
    (r"bitumen\s*felt|felt\s*waterproof",       "bitumen felt waterproofing IS 1322"),
    # ── Section 10: Plumbing / Sanitary ─────────────────────────────────────
    (r"(?:cast\s*iron|ci|ductile\s*iron|di)\s*pipe", "cast iron ductile iron pipe IS 1536 IS 8329"),
    (r"gi\s*pipe|galvanized\s*(?:iron|steel)\s*pipe", "galvanized steel pipe IS 1239"),
    (r"cpvc|chlorinated\s*pvc",                 "CPVC pipe IS 15778"),
    (r"(?:upvc|unplasticised\s*pvc)\s*pipe",    "uPVC pipe IS 4985"),
    (r"hdpe\s*pipe|pe100\s*pipe",               "HDPE pipe IS 4984"),
    (r"gi\s*(?:fitting|elbow|tee|coupling)",    "gi pipe fittings IS 1239"),
    (r"(?:ball|gate|globe|butterfly|check)\s*valve", "valve IS 778 IS 780"),
    (r"flush\s*(?:valve|cistern)|water\s*closet\s*cistern", "flushing cistern IS 774"),
    (r"wash\s*basin|lavatory\s*basin",          "wash basin IS 2556"),
    (r"water\s*(?:meter|metre)",                "water meter IS 779"),
    # ── Section 11: Hardware ─────────────────────────────────────────────────
    (r"tower\s*bolt|flush\s*bolt|aldrop",       "tower bolt IS 204 builders hardware"),
    (r"(?:butt\s*)?hinge|parliament\s*hinge",   "hinges IS 1341 door hardware"),
    (r"padlock|door\s*lock|mortice\s*lock",     "padlock IS 4991 lock IS 2209"),
    (r"door\s*(?:closer|check|spring)",         "door closer IS 3564"),
    # ── Section 12: Wood products ────────────────────────────────────────────
    (r"particle\s*board|chipboard",             "particle board IS 3087"),
    (r"(?:medium\s*density\s*)?fibre\s*?board|mdf", "fibreboard MDF IS 12406"),
    (r"block\s*board",                          "blockboard IS 1659"),
    # ── Section 13: Doors & Windows ─────────────────────────────────────────
    (r"(?:timber|wooden|panell\w+)\s*door",     "timber door shutter IS 1003"),
    (r"(?:steel|metal)\s*(?:door|window|frame)","steel door window IS 1038 IS 1361"),
    (r"(?:aluminium|aluminum)\s*(?:door|window)","aluminium door window IS 1948 IS 1949"),
    (r"rolling\s*shutter",                      "rolling shutter IS 6248"),
    (r"upvc\s*(?:door|window)",                 "uPVC door window IS 14856"),
    # ── Section 14: Reinforcement ────────────────────────────────────────────
    (r"(?:tmt|deformed|ribbed|tor|hysd)\s*(?:bar|steel|rebar)", "TMT deformed bar IS 1786 Fe 415 Fe 500"),
    (r"mild\s*steel\s*(?:bar|rod)|ms\s*bar",   "mild steel bar IS 432 concrete reinforcement"),
    (r"binding\s*wire",                         "binding wire IS 280 annealed"),
    (r"welded\s*wire\s*(?:fabric|mesh|reinforcement)", "welded wire fabric IS 1566"),
    (r"fe\s*(?:415|500|550|600)",               "HYSD deformed bar IS 1786 Fe 415 Fe 500"),
    # ── Section 15: Structural Steel ─────────────────────────────────────────
    (r"structural\s*steel|mild\s*steel\s*(?:plate|section)", "structural steel IS 2062 IS 1977"),
    (r"(?:ismb|isjb|ismc|isa|islb)\s*|steel\s*(?:angle|channel|i.beam)", "structural steel sections IS 808"),
    (r"hollow\s*(?:structural\s*)?section|hss|rhs|shs|chs", "hollow structural section IS 4923"),
    # ── Section 16: Aluminium ────────────────────────────────────────────────
    (r"wrought\s*aluminium|aluminium\s*(?:bar|rod|section|sheet|plate|tube|extrusion)", "wrought aluminium IS 733 IS 737"),
    # ── Section 19: Fasteners ────────────────────────────────────────────────
    (r"hex\s*(?:bolt|nut)|hexagonal\s*(?:bolt|nut)", "hexagonal bolt nut IS 1363 IS 1364"),
    (r"anchor\s*bolt|foundation\s*bolt",        "anchor bolt IS 5624"),
    (r"\brivet\b",                              "rivet IS 1929 IS 2998"),
    # ── Section 21: Glass ────────────────────────────────────────────────────
    (r"(?:safety|toughened|tempered|laminated|float|sheet)\s*glass", "safety glass IS 2553 toughened IS 2065"),
    (r"wired\s*glass|fire\s*resistant\s*glass", "wired glass IS 5437"),
    # ── Section 23: Insulation ───────────────────────────────────────────────
    (r"(?:rock|mineral|glass|slag)\s*wool",     "rock wool glass wool IS 3677 IS 8183 thermal insulation"),
    (r"thermal\s*insulation\s*(?:board|slab|blanket)", "thermal insulation IS 3677 IS 8183"),
    # ── Section 24: Plastics ─────────────────────────────────────────────────
    (r"phenolic\s*laminate|hpl|formica",        "phenolic laminated sheet IS 2036"),
    (r"(?:polythene|polyethylene)\s*sheet",     "polyethylene sheet IS 2508"),
    # ── Section 25: Cables ───────────────────────────────────────────────────
    (r"pvc\s*insulated\s*(?:cable|wire)|electrical\s*(?:cable|wiring)", "PVC insulated cable IS 694 IS 1554"),
    (r"armoured\s*cable|swa\s*cable",           "armoured cable IS 1554"),
    # ── Section 26: Wiring Accessories ──────────────────────────────────────
    (r"(?:modular\s*)?switch|socket\s*outlet|plug\s*(?:and\s*socket)?", "switch socket IS 3854 IS 1293"),
    (r"\bmcb\b|miniature\s*circuit\s*breaker",  "MCB IS 8828 miniature circuit breaker"),
    (r"distribution\s*board|db\s*board",        "distribution board IS 8623"),
]


def expand_query(query):
    extras = []
    q = query.lower()
    for pattern, expansion in EXPANSION_RULES:
        if re.search(pattern, q, re.IGNORECASE):
            extras.append(expansion)
    return query + " " + " ".join(extras) if extras else query


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_corpus():
    chunks = []
    for std_id, meta in STANDARDS_DB.items():
        text = (
            f"{std_id} {std_id} {std_id} {std_id} "
            f"{meta['title']} {meta['title']} "
            f"Section {meta['section']} {meta['category']} "
            f"Scope: {meta['scope']} "
        )
        chunks.append({
            "standard_id": std_id,
            "title":       meta["title"],
            "category":    meta["category"],
            "section":     meta["section"],
            "scope":       meta["scope"],
            "text":        text,
        })
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# INDEX — Hybrid BM25 + TF-IDF
# ══════════════════════════════════════════════════════════════════════════════

class BISIndex:
    def __init__(self):
        self.chunks       = []
        self.bm25         = None
        self.tfidf        = None
        self.tfidf_matrix = None
        self._built       = False

    def build(self, pdf_path=None):
        self.chunks = build_corpus()

        if pdf_path and Path(pdf_path).exists() and HAS_FITZ:
            self._enrich_from_pdf(pdf_path)

        texts         = [c["text"] for c in self.chunks]
        tokenized     = [t.lower().split() for t in texts]
        self.bm25     = BM25Okapi(tokenized)
        self.tfidf    = TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        self._built   = True
        print(f"[BISIndex] Ready — {len(self.chunks)} standards indexed across 27 sections.")

    def _enrich_from_pdf(self, pdf_path):
        import fitz as _fitz
        IS_PAT = re.compile(
            r"IS\s*(\d+)\s*(?:[\(\[]\s*(?:PART|Part)\s*(\d+)\s*[\)\]])?\s*[:\-]\s*(\d{4})",
            re.IGNORECASE
        )
        doc      = _fitz.open(pdf_path)
        enriched = 0
        for pg in range(doc.page_count):
            text   = doc[pg].get_text()
            blocks = re.split(r"SUMMARY OF\s*\n", text)
            for block in blocks[1:]:
                lines  = [l.strip() for l in block.split("\n") if l.strip()]
                if not lines: continue
                header = " ".join(lines[:6])
                m      = IS_PAT.search(header)
                if not m: continue
                num_main = m.group(1); num_part = m.group(2); year = m.group(3)
                raw_id   = f"IS {num_main} (PART {num_part}): {year}" if num_part else f"IS {num_main}: {year}"
                key      = normalise_id(raw_id)
                if not key: continue
                block_clean = re.sub(r"\s+", " ", block[:1200])
                for chunk in self.chunks:
                    if chunk["standard_id"] == key:
                        chunk["text"] += " PDF: " + block_clean
                        enriched += 1
                        break
        doc.close()
        print(f"[BISIndex] PDF enrichment: {enriched}/{len(self.chunks)} standards enriched.")

    def retrieve(self, query, top_k=5):
        expanded = expand_query(query)
        tokens   = expanded.lower().split()

        bm25_raw = np.array(self.bm25.get_scores(tokens))
        q_vec    = self.tfidf.transform([expanded])
        cos_raw  = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        def _norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-10)

        combined = 0.40 * _norm(bm25_raw) + 0.60 * _norm(cos_raw)
        top_idx  = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_idx:
            c         = dict(self.chunks[idx])
            c["score"] = float(combined[idx])
            results.append(c)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON API
# ══════════════════════════════════════════════════════════════════════════════

_index = None

def get_index(pdf_path=None):
    global _index
    if _index is None:
        _index = BISIndex()
        _index.build(pdf_path=pdf_path)
    return _index


def query_pipeline(query, top_k=5, pdf_path=None):
    t0      = time.time()
    index   = get_index(pdf_path=pdf_path)
    results = index.retrieve(query, top_k=top_k)
    latency = round(time.time() - t0, 4)
    return {
        "retrieved_standards": [r["standard_id"] for r in results],
        "results":             results,
        "latency_seconds":     latency,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST — Official public test set + cross-section tests
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pdf = "dataset.pdf" if Path("dataset.pdf").exists() else None

    TESTS = [
        ("PUB-01", "We manufacture 33 Grade Ordinary Portland Cement",                           "IS 269: 1989"),
        ("PUB-02", "Coarse and fine aggregates from natural sources for structural concrete",     "IS 383: 1970"),
        ("PUB-03", "Precast concrete pipes with and without reinforcement for water mains",       "IS 458: 2003"),
        ("PUB-04", "Hollow and solid lightweight concrete masonry blocks dimensions requirements","IS 2185 (PART 2): 1983"),
        ("PUB-05", "Corrugated and semi-corrugated asbestos cement sheets roofing cladding",     "IS 459: 1992"),
        ("PUB-06", "Portland slag cement manufacture chemical physical requirements",             "IS 455: 1989"),
        ("PUB-07", "Portland pozzolana cement calcined clay based plant",                        "IS 1489 (PART 2): 1991"),
        ("PUB-08", "Masonry cement general purpose mortars not intended for structural concrete", "IS 3466: 1988"),
        ("PUB-09", "Supersulphated cement for marine works and aggressive water conditions",      "IS 6909: 1990"),
        ("PUB-10", "White Portland cement architectural and decorative purposes",                 "IS 8042: 1989"),
        ("SEC-02", "Building limes hydraulic lime fat lime",                                     "IS 712: 1984"),
        ("SEC-04", "Common burnt clay building bricks",                                          "IS 1077: 1992"),
        ("SEC-04", "Fly ash lime bricks for masonry",                                            "IS 12894: 2002"),
        ("SEC-06", "Plywood for general purpose",                                                "IS 303: 1989"),
        ("SEC-08", "Cement concrete flooring tiles",                                             "IS 1237: 1980"),
        ("SEC-10", "Galvanized steel pipes water gas",                                           "IS 1239 (PART 1): 2004"),
        ("SEC-10", "Gate valve for water supply",                                                "IS 780: 1984"),
        ("SEC-11", "Tower bolts builders hardware",                                              "IS 204 (PART 1): 1992"),
        ("SEC-14", "TMT deformed steel bar Fe 415 Fe 500 reinforcement",                        "IS 1786: 2008"),
        ("SEC-15", "Hot rolled structural steel plates sections",                                "IS 2062: 2006"),
        ("SEC-21", "Toughened safety glass for general purpose",                                 "IS 2553 (PART 1): 1990"),
        ("SEC-23", "Rock wool thermal insulation slabs",                                         "IS 3677: 1985"),
        ("SEC-25", "PVC insulated cables wiring",                                                "IS 694: 2010"),
        ("SEC-26", "Plugs and socket outlets 250V wiring accessories",                           "IS 1293: 2005"),
    ]

    print("\n" + "="*72)
    print("  BIS RAG — FULL 27-SECTION SELF TEST")
    print("="*72)

    hits3 = 0; mrr_sum = 0.0
    for tid, query, expected in TESTS:
        r     = query_pipeline(query, top_k=5, pdf_path=pdf)
        top5  = r["retrieved_standards"]
        top3  = top5[:3]
        ne    = re.sub(r"\s+", "", expected).upper()
        nt5   = [re.sub(r"\s+", "", x).upper() for x in top5]
        hit   = ne in [re.sub(r"\s+", "", x).upper() for x in top3]
        hits3 += int(hit)
        rank  = next((i+1 for i, s in enumerate(nt5) if s == ne), None)
        mrr   = 1.0 / rank if rank else 0.0
        mrr_sum += mrr
        status = "+" if hit else "-"
        print(f"  [{status}] [{tid}] expected={expected:<38} got={top3[0]}")

    n = len(TESTS)
    print(f"\n  Hit Rate @3 : {hits3}/{n} = {100*hits3/n:.1f}%")
    print(f"  MRR @5      : {mrr_sum/n:.4f}")
    print("="*72)
