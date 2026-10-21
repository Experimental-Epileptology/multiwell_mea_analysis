# Platebook

Both date fields accept `DDMMYY` (e.g. `030425`), `DDMMYYYY`, `YYYYMMDD`, separated dates such as `03.04.2025` or `2025-04-03`, and English/German month names such as `3 April 2025`. Ambiguous numeric dates use day-first order; two-digit years mean 2000–2099. Clearly month-first separated dates such as `12/31/2025` are also recognized. Dates are checked against the calendar and normalized to `YYYY-MM-DD`. Review shows the interpreted dates before appending.

Use the per-well **Exclude** grid to choose Yes or No (default No). Excluded wells are retained in appended metadata and CSV exports with an `exclude` column; filter this column during analysis. Put the exclusion reason in Notes. Older projects receive No for this field.

Open `index.html` in a modern Chrome, Edge, Firefox, or Safari browser. No installation, server, account, or internet connection is needed. Keep the HTML file to run the app on another computer.

1. Enter the recording name (required before append), experiment and physical plate IDs, default time and units. The default recording name is used unless a well has its own override.
2. Select a field such as Cell line or Treatment. Type into the 4 × 6 grid, use Fill all wells, or paste a rectangular Excel block starting in a well. Repeat for each field.
3. Enter a cell line, optional treatment, optional numeric dose, dose unit and time or time label for every occupied well. Empty time/unit overrides use the shared defaults. Use `n.a.` as a unit when appropriate. Blank-cell-line wells with other entries are flagged, not silently discarded.

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

Each append can have a new recording name. Enter a shared Default recording name, or use the Recording name override plate grid for individual wells. Every occupied well needs a resolved name. Review shows each name. After appending, all recording-name inputs clear while other conditions stay filled; previous records retain their names. Enter the next timepoint and a fresh name for the next append.

## Custom fields

Click **Add custom field**, enter a unique name, choose **All wells (shared)** or **Individual wells (plate grid)**, and select Text, Number, Date or Dropdown. For a dropdown, enter comma-separated choices. Standard fields remain available. Custom fields are optional and belong to the saved project.

Shared inputs appear under Plate details; per-well fields appear as additional tabs and support the same paste/fill actions. Number/date/dropdown entries are validated before appending. Dates use the same flexible formats as standard dates. Review shows the custom values.

Each field adds a metadata column. CSV uses stable keys such as `custom_medium`; the on-screen table uses the field name. Values are saved with each appended record, so changing a shared value only affects future appends. Older records get blanks when a new field is added. Field definitions and input values travel with Save project / Open project. Older projects open with no custom fields.

New plate clears per-well custom entries and keeps shared custom inputs. Clear plate form clears custom inputs but preserves definitions and appended records. Custom field definitions cannot yet be renamed or removed. Up to 100 custom fields are supported.
