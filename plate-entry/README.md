# Platebook

Use the per-well **Exclude** grid to choose Yes or No (default No). Excluded wells are retained in appended metadata and CSV exports with an `exclude` column; filter this column during analysis. Put the exclusion reason in Notes. Older projects receive No for this field.

Open `index.html` in a modern Chrome, Edge, Firefox, or Safari browser. No installation, server, account, or internet connection is needed. Keep the HTML file to run the app on another computer.

1. Enter the recording name (required before append), experiment and physical plate IDs, default time and units. The recording name is copied to every appended well.
2. Select a field such as Cell line or Treatment. Type into the 4 × 6 grid, use Fill all wells, or paste a rectangular Excel block starting in a well. Repeat for each field.
3. Enter a cell line, treatment, optional numeric dose, dose unit and time or time label for every occupied well. Empty time/unit overrides use the shared defaults. Use `n.a.` as a unit when appropriate. Blank-cell-line wells with other entries are flagged, not silently discarded.

   Batch, DIV (days in vitro), differentiation date and recording date have their own plate grids and support paste and Fill all wells. These fields are optional. DIV must be a non-negative whole number when supplied; dates accept `YYYY-MM-DD` or `YYYYMMDD` and are exported as `YYYY-MM-DD`. DIV is recorded manually, not calculated from differentiation date. Older projects open with these new fields blank and retain their existing records.
4. Choose Review & append, inspect the records, then Append records.
5. For another repeated timepoint, leave the well time overrides blank and change the shared time. For destructive sampling, use different physical wells or plates for each collection.
6. Save project downloads the complete editable project as JSON. Open project replaces the current project after confirmation. Export CSV downloads the accumulated metadata for Excel or analysis tools.

Browser storage is a convenience, not a backup. It can be unavailable for local files or be cleared by browser settings. Save a project file regularly and use it to transfer data between computers. This app does not sync or send data to a server. Avoid editing the same project on multiple computers at once; merging projects is not supported.

Sample IDs encode [plate ID, well, time or text label] as a JSON array to avoid ambiguous separator collisions. Preserve IDs once used in result files. CSV quotes fields and prefixes spreadsheet formula-like text with an apostrophe to keep it as text; project JSON preserves original strings.

The example is fictional. Timepoints are not biological replicates. Plate allocation and randomization are decisions for your experiment.

Validation rejects duplicate plate–well–time combinations and changes to cell line or experiment on an existing physical well. Treatment can change over time; record the treatment relevant to each observation and describe changes in Notes.

The current version appends records without editing or deleting earlier entries. Review entries before appending and save a project checkpoint before each batch. To correct an appended record, edit a copy of the exported CSV for analysis, retaining the original project as the record of entry.

**New plate** clears the well-entry grids and Plate ID, keeping shared experiment details and all appended metadata. Enter a unique physical Plate ID for the new plate.

**Clear metadata table** removes all appended records after confirmation and keeps the current plate form. Save a project or CSV first if you need those records; clearing cannot be undone.
