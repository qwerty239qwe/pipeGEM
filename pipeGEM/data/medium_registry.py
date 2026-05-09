from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class MediumInfo:
    """Metadata for a named medium in the catalog.

    Attributes
    ----------
    name : str
        Filename stem (without .tsv) used to locate the medium file.
    description : str
        Short human-readable description.
    source : str
        Literature citation or derivation note.
    organism : str
        Intended host organism(s).
    medium_type : str
        One of ``'minimal'``, ``'defined'``, ``'rich'``, or ``'complex'``.
    default_id_col : str
        Column label in the TSV that contains metabolite IDs (default ``'BiGG'``).
    is_approximate : bool
        ``True`` for undefined/complex media whose composition is estimated.
    source_url : str
        URL for the original citation or source publication, when known.
    composition_url : str
        URL for the formulation used to build or validate the bundled TSV.
    composition_note : str
        Short note describing whether the TSV is an exact defined recipe,
        an ionized exchange representation, or an approximate proxy.
    """

    name: str
    description: str
    source: str
    organism: str
    medium_type: str
    default_id_col: str = "BiGG"
    is_approximate: bool = False
    source_url: str = ""
    composition_url: str = ""
    composition_note: str = ""


class MediumCatalog(Enum):
    """Catalog of named media bundled with pipeGEM.

    Each member maps to a :class:`MediumInfo` instance that carries metadata
    and the TSV filename.  Use :meth:`~pipeGEM.data.MediumData.from_catalog`
    to load a medium by catalog entry.

    Examples
    --------
    >>> from pipeGEM.data import MediumCatalog
    >>> MediumCatalog.M9.value.description
    'M9 minimal salts medium with glucose'
    >>> MediumCatalog.LB.value.is_approximate
    True
    """

    # Mammalian media.
    DMEM = MediumInfo(
        "DMEM",
        "Dulbecco's Modified Eagle Medium",
        "Eagle 1959",
        "Mammalian",
        "defined",
        default_id_col="human_1",
        source_url="https://doi.org/10.1126/science.122.3168.501",
        composition_url="https://doi.org/10.1126/science.122.3168.501",
        composition_note="Defined recipe represented as exchange metabolites.",
    )
    DMEM_HIGH_FFA = MediumInfo(
        "DMEM_high_FFA",
        "DMEM with elevated free fatty acids",
        "Derived from DMEM",
        "Mammalian",
        "defined",
        default_id_col="human_1",
        source_url="https://doi.org/10.1126/science.122.3168.501",
        composition_url="https://doi.org/10.1126/science.122.3168.501",
        composition_note="Derived DMEM recipe with added free fatty acids.",
    )
    HAMS = MediumInfo(
        "Hams",
        "Ham's F-12 Nutrient Mixture",
        "Ham 1965",
        "Mammalian",
        "defined",
        default_id_col="human_1",
        source_url="https://doi.org/10.1073/pnas.53.2.288",
        composition_url="https://doi.org/10.1073/pnas.53.2.288",
        composition_note="Defined recipe represented as exchange metabolites.",
    )
    SERUM = MediumInfo(
        "serum",
        "Serum-based medium (approximate)",
        "Ham & McKeehan 1979",
        "Mammalian",
        "complex",
        default_id_col="human_1",
        is_approximate=True,
        source_url="https://doi.org/10.1016/0076-6879(79)58105-0",
        composition_url="https://doi.org/10.1016/0076-6879(79)58105-0",
        composition_note="Approximate serum metabolite proxy, not a fixed defined medium recipe.",
    )

    # Bacterial minimal / defined media.
    M9 = MediumInfo(
        "M9",
        "M9 minimal salts medium with glucose",
        "Miller 1972",
        "E. coli, Gram-negatives",
        "minimal",
        source_url="https://mediadive.dsmz.de/medium/382",
        composition_url="https://mediadive.dsmz.de/medium/382",
        composition_note="Defined salts recipe decomposed into exchange metabolites.",
    )
    M63 = MediumInfo(
        "M63",
        "M63 minimal medium with glucose",
        "Pardee et al. 1959",
        "E. coli, Gram-negatives",
        "minimal",
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5752254/",
        composition_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5752254/",
        composition_note="Defined salts recipe decomposed into exchange metabolites.",
    )
    MOPS = MediumInfo(
        "MOPS",
        "MOPS defined minimal medium",
        "Neidhardt et al. 1974",
        "E. coli",
        "minimal",
        source_url="https://doi.org/10.1128/jb.119.3.736-747.1974",
        composition_url="https://doi.org/10.1128/jb.119.3.736-747.1974",
        composition_note="Defined salts recipe decomposed into exchange metabolites.",
    )
    CGXII = MediumInfo(
        "CGXII",
        "CGXII minimal medium for C. glutamicum",
        "Keilhauer et al. 1993",
        "C. glutamicum",
        "minimal",
        source_url="https://doi.org/10.1099/00221287-139-3-555",
        composition_url="https://doi.org/10.1099/00221287-139-3-555",
        composition_note="Defined salts recipe decomposed into exchange metabolites.",
    )

    # Bacterial rich / complex media.
    LB = MediumInfo(
        "LB",
        "Lysogeny Broth (approximate composition)",
        "Bertani 1951",
        "E. coli",
        "rich",
        is_approximate=True,
        source_url="https://doi.org/10.1128/jb.62.3.293-300.1951",
        composition_url="https://actinobase.org/index.php/LB",
        composition_note="Approximate amino-acid proxy for tryptone and yeast extract; salts and glucose are explicit.",
    )
    TB = MediumInfo(
        "TB",
        "Terrific Broth (approximate composition)",
        "Tartof & Hobbs 1987",
        "E. coli",
        "rich",
        is_approximate=True,
        source_url="https://doi.org/10.1038/350016a0",
        composition_url="https://www.himedialabs.com/us/fields-of-application/industry/molecular-biology/m1250-tartoff-hobbs-broth-terrific-broth.html",
        composition_note="Approximate amino-acid proxy for tryptone and yeast extract; glycerol and phosphate buffer are explicit.",
    )
    BHI = MediumInfo(
        "BHI",
        "Brain Heart Infusion (approximate composition)",
        "Rosenow 1919",
        "Fastidious organisms",
        "complex",
        is_approximate=True,
        source_url="https://www.thermofisher.com/order/catalog/product/R452472",
        composition_url="https://www.fda.gov/food/laboratory-methods/bam-media-m24-brain-heart-infusion-bhi-broth-and-agar",
        composition_note="Approximate amino-acid proxy for brain/heart infusion and peptone; glucose and salts are explicit.",
    )
    R2A = MediumInfo(
        "R2A",
        "Reasoner's 2A low-nutrient agar medium (approximate composition)",
        "Reasoner & Geldreich 1985",
        "Environmental isolates",
        "complex",
        is_approximate=True,
        source_url="https://doi.org/10.1128/aem.49.1.1-7.1985",
        composition_url="https://mediadive.dsmz.de/medium/830",
        composition_note="Approximate amino-acid proxy for yeast extract, proteose peptone, and casamino acids; defined carbon and salts are explicit.",
    )
    SOC = MediumInfo(
        "SOC",
        "SOC recovery medium (approximate composition)",
        "Hanahan 1983",
        "E. coli",
        "rich",
        is_approximate=True,
        source_url="https://doi.org/10.1016/S0076-6879(83)01016-6",
        composition_url="https://www.laboratorynotes.com/preparation-of-soc-super-optimal-broth-with-catabolite-repression-medium/",
        composition_note="Approximate amino-acid proxy for tryptone and yeast extract; glucose, magnesium, and salts are explicit.",
    )
