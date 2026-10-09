use super::*;
use arrow::array::Array;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub(super) struct Table {
    pub(super) rows: Vec<BTreeMap<String, String>>,
}
impl Table {
    pub(super) fn has(&self, name: &str) -> bool {
        self.rows[0].contains_key(name)
    }
    pub(super) fn strings(&self, name: &str) -> Result<Vec<String>> {
        self.rows
            .iter()
            .map(|r| {
                r.get(name)
                    .cloned()
                    .with_context(|| format!("missing column {name}"))
            })
            .collect()
    }
    pub(super) fn numbers(&self, name: &str) -> Result<Vec<f64>> {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let s = r
                    .get(name)
                    .with_context(|| format!("missing column {name}"))?;
                let v = s
                    .parse::<f64>()
                    .with_context(|| format!("row {}: {name} is not numeric", i + 1))?;
                ensure!(v.is_finite(), "row {}: {name} is not finite", i + 1);
                Ok(v)
            })
            .collect()
    }
}

pub(super) fn load(path: &Path, columns: &BTreeMap<String, String>) -> Result<Table> {
    let mut rows = Vec::new();
    if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
        let builder = ParquetRecordBatchReaderBuilder::try_new(fs::File::open(path)?)?;
        let indices = columns
            .iter()
            .map(|(name, source)| Ok((name.clone(), builder.schema().index_of(source)?)))
            .collect::<Result<Vec<_>>>()?;
        // Project only declared columns, including metric columns; no accidental label scans.
        let projection = parquet::arrow::ProjectionMask::roots(
            builder.parquet_schema(),
            indices.iter().map(|(_, i)| *i),
        );
        let reader = builder
            .with_projection(projection)
            .with_batch_size(4096)
            .build()?;
        for batch in reader {
            let batch = batch?;
            for row in 0..batch.num_rows() {
                let mut mapped = BTreeMap::new();
                for (name, source) in columns {
                    let array = batch.column(batch.schema().index_of(source)?);
                    let value = if array.is_null(row) {
                        String::new()
                    } else {
                        arrow::util::display::array_value_to_string(array.as_ref(), row)?
                    };
                    mapped.insert(name.clone(), value);
                }
                rows.push(mapped);
            }
        }
    } else {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        ensure!(
            matches!(ext, "csv" | "tsv"),
            "expected CSV, TSV or Parquet: {}",
            path.display()
        );
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(if ext == "csv" { b',' } else { b'\t' })
            .from_path(path)?;
        let headers = reader.headers()?.clone();
        ensure!(
            headers.iter().collect::<BTreeSet<_>>().len() == headers.len(),
            "duplicate table headers"
        );
        let indices = columns
            .iter()
            .map(|(name, source)| {
                Ok((
                    name.clone(),
                    headers
                        .iter()
                        .position(|h| h == source)
                        .with_context(|| format!("missing column {source}"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        for record in reader.records() {
            let record = record?;
            rows.push(
                indices
                    .iter()
                    .map(|(name, i)| (name.clone(), record[*i].to_string()))
                    .collect(),
            );
        }
    }
    ensure!(!rows.is_empty(), "dataset has no rows");
    let mut ids = BTreeSet::new();
    for row in &rows {
        let id = &row["id"];
        ensure!(
            !id.is_empty() && ids.insert(id),
            "empty or duplicate row id: {id:?}"
        );
    }
    Ok(Table { rows })
}

pub(super) fn write_scores(
    path: &Path,
    table: &Table,
    scores: &[super::adapter::Score],
) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = csv::WriterBuilder::new().delimiter(b'\t').from_writer(file);
    writer.write_record(["id", "score", "error", "provenance"])?;
    for (row, s) in table.rows.iter().zip(scores) {
        writer.write_record([
            row["id"].clone(),
            s.value.map(|v| v.to_string()).unwrap_or_default(),
            s.error.clone().unwrap_or_default(),
            s.provenance.to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}
