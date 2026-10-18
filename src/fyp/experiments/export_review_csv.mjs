// Author a plain, editable CSV through the bundled Artifact Tool value grid.
// No formulas, formatting or row truncation; every source field round-trips.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [inputPath, outputPath, nodeModules] = process.argv.slice(2);
// Read-only package resolution avoids needing a filesystem junction/elevation.
const resolveBundled = createRequire(path.join(nodeModules, '..', 'review-resolver.cjs'));
const { Workbook } = await import(pathToFileURL(resolveBundled.resolve('@oai/artifact-tool')).href);
const rows = JSON.parse(await fs.readFile(inputPath, 'utf8'));
const headers = Object.keys(rows[0]);
const matrix = [headers, ...rows.map(r => headers.map(k => String(r[k])))];
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('Human review');
const range = sheet.getRangeByIndexes(0, 0, matrix.length, headers.length);
range.setNumberFormat('@');
range.values = matrix;
workbook.recalculate();
const values = range.values.map(row => row.map(v => v === null ? '' : String(v)));
if (JSON.stringify(values) !== JSON.stringify(matrix)) throw new Error('Artifact value grid changed source text');
const inspection = await workbook.inspect({kind:'region',sheetId:sheet.name,range:'A1:C3',maxChars:1000,tableMaxRows:3,tableMaxCols:3});
console.log(inspection.ndjson);
const quote = value => '"' + value.replaceAll('"','""') + '"';
// CSV has no visual formatting. The companion Markdown provides the reading layout.
await fs.writeFile(outputPath, '\uFEFF' + values.map(row => row.map(quote).join(',')).join('\r\n') + '\r\n', {encoding:'utf8',flag:'wx'});
console.log(`Saved ${rows.length} paired review rows; ${headers.length} columns.`);
