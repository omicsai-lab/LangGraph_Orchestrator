#!/usr/bin/env Rscript
# Independent R DESeq2 reference: no Python-derived results used for fitting.
# Run from repository root: Rscript validation/deseq2_reference.R

options(stringsAsFactors = FALSE)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 0L && length(args) != 3L) {
  stop("Usage: Rscript validation/deseq2_reference.R [counts.csv metadata.csv output_dir]", call. = FALSE)
}
counts_path <- if (length(args)) args[[1]] else "bc_counts_transposed_condensed.csv"
metadata_path <- if (length(args)) args[[2]] else "bc_meta.csv"
outdir <- if (length(args)) args[[3]] else "benchmark_outputs/v2_3b_r_reference"

if (!requireNamespace("DESeq2", quietly = TRUE)) stop("Install DESeq2 before running.")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install jsonlite before running.")

# Read as character, never permit automatic header repairs or inferred count types.
read_csv_strict <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  x <- utils::read.csv(path, check.names = FALSE, colClasses = "character",
                       na.strings = NULL, fileEncoding = "UTF-8-BOM", strip.white = FALSE)
  header <- names(x)
  if (nrow(x) == 0L || ncol(x) < 2L) stop("Empty/insufficient CSV: ", path)
  if (anyNA(header) || any(!nzchar(trimws(header))) || anyDuplicated(header))
    stop("Invalid or duplicate CSV headers: ", path)
  x
}
raw_counts <- read_csv_strict(counts_path)
raw_meta <- read_csv_strict(metadata_path)
if (!all(c("Sample", "condition") %in% names(raw_meta)))
  stop("Metadata requires 'Sample' and 'condition' columns")
if (anyNA(raw_meta$Sample) || any(!nzchar(trimws(raw_meta$Sample))) || anyDuplicated(raw_meta$Sample))
  stop("Missing/duplicate sample identifiers in metadata")
if (anyNA(raw_counts[[1]]) || any(!nzchar(trimws(raw_counts[[1]]))) || anyDuplicated(raw_counts[[1]]))
  stop("Missing/duplicate first-column identifiers in counts")

sample_ids <- raw_meta$Sample
row_ids <- raw_counts[[1]]
feature_headers <- names(raw_counts)[-1]
row_match <- length(sample_ids) == length(row_ids) && setequal(sample_ids, row_ids)
col_match <- length(sample_ids) == length(feature_headers) && setequal(sample_ids, feature_headers)
if (identical(row_match, col_match))
  stop("Counts orientation ambiguous or sample identifiers do not match exactly")

# All values must be present, finite, nonnegative, integer-valued and safely representable.
values <- as.matrix(raw_counts[-1])
if (anyNA(values) || any(!nzchar(trimws(values)))) stop("Missing/blank count values")
parsed <- suppressWarnings(matrix(as.numeric(values), nrow = nrow(values), dimnames = dimnames(values)))
if (anyNA(parsed) || any(!is.finite(parsed)) || any(parsed < 0) || any(parsed != floor(parsed)))
  stop("Counts contain nonnumeric, nonfinite, negative or fractional values")
if (any(parsed > .Machine$integer.max)) stop("Counts exceed R integer range")

if (row_match) {
  rownames(parsed) <- row_ids
  colnames(parsed) <- feature_headers
  samples_by_genes <- parsed[sample_ids, , drop = FALSE]
  orientation <- "samples_by_genes"
} else {
  rownames(parsed) <- row_ids
  colnames(parsed) <- feature_headers
  samples_by_genes <- t(parsed[, sample_ids, drop = FALSE])
  orientation <- "genes_by_samples"
}
if (anyDuplicated(colnames(samples_by_genes))) stop("Duplicate gene identifiers")

numerator <- "risk category: High"
reference <- "risk category: Average"
if (anyNA(raw_meta$condition) || any(!nzchar(trimws(raw_meta$condition))) ||
    !setequal(unique(raw_meta$condition), c(numerator, reference)))
  stop("Unexpected condition levels or missing conditions")
meta <- raw_meta[match(rownames(samples_by_genes), raw_meta$Sample), , drop = FALSE]
rownames(meta) <- meta$Sample
stopifnot(identical(rownames(samples_by_genes), rownames(meta)))
if (any(table(meta$condition) < 2L)) stop("Insufficient group replication")

# Same explicit prefilter as Python: >=10 reads in >=ceil(20% of samples).
min_count <- 10L
min_fraction <- 0.20
min_samples <- max(1L, as.integer(ceiling(nrow(samples_by_genes) * min_fraction)))
keep <- colSums(samples_by_genes >= min_count) >= min_samples
filtered <- samples_by_genes[, keep, drop = FALSE]
if (ncol(filtered) == 0L) stop("No genes retained")
# DESeq2 expects genes x samples; raw input and filtered input remain separate.
count_matrix <- t(filtered)
storage.mode(count_matrix) <- "integer"
meta$condition <- factor(meta$condition, levels = c(reference, numerator))

set.seed(42)
start <- proc.time()[["elapsed"]]
dds <- DESeq2::DESeqDataSetFromMatrix(countData = count_matrix,
                                    colData = meta, design = ~ condition)
dds <- DESeq2::DESeq(dds, fitType = "parametric", minReplicatesForReplace = 7,
                    parallel = FALSE, quiet = FALSE)
res <- DESeq2::results(dds, contrast = c("condition", numerator, reference),
                       alpha = 0.05, independentFiltering = TRUE,
                       cooksCutoff = TRUE)
elapsed <- proc.time()[["elapsed"]] - start

result <- as.data.frame(res)[, c("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"), drop = FALSE]
result <- data.frame(gene_id = rownames(result), result, check.names = FALSE)

# Capture actual dispersion fit method and outlier replacement diagnostics where available.
fit_type <- attr(DESeq2::dispersionFunction(dds), "fitType")
if (is.null(fit_type)) fit_type <- NA_character_
assay_names <- SummarizedExperiment::assayNames(dds)
replaced_assay <- "replaceCounts" %in% assay_names
replaced_genes <- if (replaced_assay) {
  before <- SummarizedExperiment::assay(dds, "counts", withDimnames = TRUE)
  after <- SummarizedExperiment::assay(dds, "replaceCounts", withDimnames = TRUE)
  rownames(before)[rowSums(before != after) > 0L]
} else character(0)

# Source hashes computed with tools::md5sum for R provenance; SHA-256 is not inferred.
# The comparison uses exact source file paths and independent filtering decisions.
manifest <- list(
  run_type = "independent_R_DESeq2_reference",
  counts_path = normalizePath(counts_path, winslash = "/"),
  metadata_path = normalizePath(metadata_path, winslash = "/"),
  counts_md5 = unname(tools::md5sum(counts_path)),
  metadata_md5 = unname(tools::md5sum(metadata_path)),
  r_version = R.version.string,
  deseq2_version = as.character(utils::packageVersion("DESeq2")),
  bioconductor_version = as.character(BiocManager::version()),
  samples = nrow(samples_by_genes), genes_input = ncol(samples_by_genes),
  orientation = orientation,
  group_sizes = as.list(table(meta$condition)),
  filter_audit = list(min_count = min_count, min_fraction = min_fraction,
                      min_samples = min_samples, genes_before = ncol(samples_by_genes),
                      genes_after = ncol(filtered), excluded_genes = colnames(samples_by_genes)[!keep],
                      retained_genes = colnames(filtered)),
  model = list(design = "~ condition", numerator = numerator, reference = reference,
               fit_type_requested = "parametric", fit_type_used = fit_type,
               minReplicatesForReplace = 7L, cooksCutoff = TRUE,
               independentFiltering = TRUE, alpha = 0.05, elapsed_seconds = elapsed,
               replace_counts_assay_present = replaced_assay,
               replaced_genes = replaced_genes),
  results_rows = nrow(result),
  missing_padj = sum(is.na(result$padj)),
  missing_pvalue = sum(is.na(result$pvalue))
)

# Only create outputs after the full validation and fit have succeeded.
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(result, file.path(outdir, "deseq2_results.csv"), row.names = FALSE, na = "NA", quote = TRUE)
utils::write.csv(data.frame(gene_id = colnames(samples_by_genes), retained = as.logical(keep)),
                 file.path(outdir, "filtering_audit.csv"), row.names = FALSE)
jsonlite::write_json(manifest, file.path(outdir, "model_diagnostics.json"),
                     pretty = TRUE, auto_unbox = TRUE, null = "null", na = "null")
writeLines(capture.output(sessionInfo()), file.path(outdir, "session_info.txt"))
cat("R DESeq2 reference completed; results:", nrow(result), "genes; time:", round(elapsed, 2), "seconds\n")
cat("Dispersion trend used:", fit_type, "; replaced genes:", length(replaced_genes), "\n")
cat("Output directory:", outdir, "\n")
