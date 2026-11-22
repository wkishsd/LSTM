// Companion XLSX exporter. Calculations and exchange collection live in Python.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const root = fileURLToPath(new URL("../../", import.meta.url));


const runtimeRequire = createRequire(import.meta.url);
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href);
const inputDir = path.resolve(process.argv[2] || path.join(root, 'Exports', 'macd_30m'));
const payload = JSON.parse(await fs.readFile(path.join(inputDir, 'workbook_data.json'), 'utf8'));
const meta = payload.summary;
const reportDir = inputDir;
const outputPath = path.join(reportDir, `macd_30m_${meta.analysis_native_candles_requested}.xlsx`);

const wb = Workbook.create();
const summary = wb.worksheets.add('Summary');
const monthly = wb.worksheets.add('Monthly');
const cycles = wb.worksheets.add('Cycles');
const hist = wb.worksheets.add('Histogram');
const ink = '#203343', header = '#294A63', muted = '#61707B';
for (const sheet of [summary, monthly, cycles, hist]) sheet.showGridLines = false;
summary.tabColor = header;

function serialDate(value) {
  // Excel dates have no timezone. Preserve the already converted UTC+3 wall time.
  return Date.parse(value.replace(' ', 'T') + ':00Z') / 86400000 + 25569;
}
function columnName(index) {
  let n = index + 1, out = '';
  while (n) { n--; out = String.fromCharCode(65 + n % 26) + out; n = Math.floor(n / 26); }
  return out;
}
function setHeader(sheet, row, labels) {
  const range = sheet.getRangeByIndexes(row - 1, 0, 1, labels.length);
  range.values = [labels];
  range.format = {fill: header, font: {name: 'Arial', size: 10, bold: true, color: '#FFFFFF'},
    wrapText: true, horizontalAlignment: 'center', verticalAlignment: 'center', rowHeight: 38};
}
function addData(sheet, source, selectedColumns, labels, title) {
  sheet.getRange('A1').values = [[title]];
  sheet.getRange('A1').format.font = {name: 'Arial', size: 14, bold: true, color: ink};
  const sourceIndices = selectedColumns.map(c => source.columns.indexOf(c));
  if (sourceIndices.some(i => i < 0)) throw new Error(`Missing column in ${title}`);
  const rows = source.rows.map(row => sourceIndices.map((idx, j) => {
    const value = row[idx];
    return source.date_columns.includes(selectedColumns[j]) && value !== null ? serialDate(value) : value;
  }));
  const last = rows.length + 4;
  sheet.getRangeByIndexes(3, 0, rows.length + 1, labels.length).format.font = {name: 'Arial', size: 10, color: ink};
  sheet.getRangeByIndexes(4, 0, rows.length, labels.length).format.rowHeight = 18;
  setHeader(sheet, 4, labels);
  for (let offset = 0; offset < rows.length; offset += 5000) {
    const block = rows.slice(offset, offset + 5000);
    sheet.getRangeByIndexes(offset + 4, 0, block.length, labels.length).values = block;
  }
  sheet.getRangeByIndexes(4, 0, rows.length, labels.length).setNumberFormat('#,##0');
  for (let j = 0; j < selectedColumns.length; j++) {
    const letter = columnName(j), col = selectedColumns[j];
    const range = sheet.getRange(`${letter}5:${letter}${last}`);
    sheet.getRange(`${letter}4:${letter}${last}`).format.columnWidth = source.date_columns.includes(col) ? 24 : 15;
    if (source.date_columns.includes(col)) range.setNumberFormat('yyyy-mm-dd hh:mm');
    if (['macd', 'signal', 'hist', 'start_hist', 'end_hist'].includes(col)) range.setNumberFormat('0.00000000');
    if (col === 'close') range.setNumberFormat('#,##0.0000');
    if (col === 'elapsed_hours') range.setNumberFormat('0.0');
    if (col === 'hist_color') range.format.horizontalAlignment = 'left';
  }
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(2);
  const table = sheet.tables.add(`A4:${columnName(labels.length - 1)}${last}`, true, `${sheet.name}Data`);
  table.showFilterButton = true;
  return {last, count: rows.length};
}

const c = addData(cycles, payload.cycles,
  ['cycle_no','start_open_utc3','end_open_utc3','start_available_utc3','end_available_utc3',
   'start_candle_no','end_candle_no','cycle_candles','strictly_between_candles','window_including_both_starts',
   'red_halfwave_candles','green_halfwave_candles','elapsed_hours','start_hist','end_hist','at_or_below_p90'],
  ['Cycle','Previous red start UTC+3','Current red start UTC+3','Previous confirmation UTC+3','Current confirmation UTC+3',
   'Start candle no.','End candle no.','Cycle candles','Strictly between','Bars including both starts',
   'Red half-wave candles','Green half-wave candles','Hours','Start hist','End hist','Fits P90'],
  'Consecutive green-to-red cycles');
cycles.getRange('A2').values = [['Length is the difference between red-start candle numbers. Both starts must be inside the analysis range.']];
cycles.getRange('A2').format.font = {name: 'Arial', size: 10, italic: true, color: muted};
cycles.getRange(`H5:J${c.last}`).formulasR1C1 = Array.from({length: c.count}, () => ['=RC[-1]-RC[-2]','=RC[-1]-1','=RC[-2]+1']);
cycles.getRange(`M5:M${c.last}`).formulasR1C1 = Array.from({length: c.count}, () => ['=RC[-5]/2']);
cycles.getRange(`P5:P${c.last}`).formulasR1C1 = Array.from({length: c.count}, () => ['=IF(RC[-8]<=Summary!R7C2,1,0)']);
cycles.getRange(`P5:P${c.last}`).conditionalFormats.add('cellIs', {operator: 'equal', formula: 0, format: {font: {color: '#A23636'}}});
console.log(`Cycle detail ready: ${c.count} completed cycles`);

const h = addData(hist, payload.candles,
  ['candle_no','open_utc3','close','macd','signal','hist','hist_color','green_to_red',
   'completed_cycle_candles','candles_since_red_start','is_synthetic'],
  ['Candle no.','Candle open UTC+3','Close','MACD 12/26','Signal 9','Histogram','Half-wave colour',
   'Green to red','Completed cycle candles','Candles since red start','Synthetic slot'],
  `${meta.symbol} native 30m MACD`);
hist.getRange('A2').values = [[`Source: ${meta.source_endpoint}`]];
hist.getRange('A2').format.font = {name: 'Arial', size: 10, italic: true, color: muted};
hist.getRange(`A4:A${h.last}`).format.columnWidth = 11;
hist.getRange(`D4:F${h.last}`).format.columnWidth = 18;
hist.getRange(`F5:F${h.last}`).conditionalFormats.add('cellIs', {operator: 'lessThan', formula: 0, format: {font: {color: '#A23636'}}});
hist.getRange(`F5:F${h.last}`).conditionalFormats.add('cellIs', {operator: 'greaterThan', formula: 0, format: {font: {color: '#21724E'}}});
console.log(`Histogram detail ready: ${h.count} chronological 30m rows`);

summary.getRange('A2').values = [[`${meta.symbol} 30m MACD cycle lengths`]];
summary.getRange('A2').format.font = {name: 'Arial', size: 14, bold: true, color: ink};
summary.getRange('A4:C30').format.font = {name: 'Arial', size: 10, color: ink};
summary.getRange('A4:C4').values = [['Measure','Result','Unit']];
summary.getRange('A4:C4').format = {fill: header, font: {name: 'Arial', size: 10, bold: true, color: '#FFFFFF'}, rowHeight: 24};
const measures = [
  ['Percentile', meta.percentile / 100, ''],
  ['Completed red-start cycles', null, 'cycles'],
  ['Minimum interval covering 90%', null, '30m periods'],
  ['Equivalent duration', null, 'hours'],
  ['Window including both red starts', null, 'candles'],
  ['Cycles covered at this length', null, 'share'],
  ['Coverage with one fewer period', null, 'share'],
  ['Interpolated 90th percentile', meta.p90_linear_interpolated, '30m periods'],
  ['Minimum cycle', null, '30m periods'],
  ['Median cycle', null, '30m periods'],
  ['Mean cycle', null, '30m periods'],
  ['Maximum cycle', null, '30m periods'],
  ['',null,''],
  ['Actual API candles analyzed',meta.native_candles_in_analysis,'candles'],
  ['Missing 30m slots filled',meta.synthetic_slots_in_analysis,'slots'],
  ['Total analyzed timeline',meta.analysis_candles,'30m slots'],
  ['Earlier MACD warm-up',meta.warmup_slots_used,'30m slots'],
  ['First candle open UTC+3',serialDate(new Date(Date.parse(meta.first_open_utc) + 3*3600000).toISOString().slice(0,16).replace('T',' ')), ''],
  ['Last candle open UTC+3',serialDate(new Date(Date.parse(meta.last_open_utc) + 3*3600000).toISOString().slice(0,16).replace('T',' ')), ''],
  ['Last candle closed UTC+3',serialDate(new Date(Date.parse(meta.last_available_utc) + 3*3600000).toISOString().slice(0,16).replace('T',' ')), ''],
  ['Excluded leading partial cycle',meta.leading_candles_before_first_crossing,'slots'],
  ['Excluded unfinished final cycle',meta.incomplete_tail_candles,'slots'],
  ['MACD / signal periods',`${meta.macd_fast} / ${meta.macd_slow} / ${meta.macd_signal}`,''],
  ['1m API candles fetched',meta.native_1m_candles_fetched,'candles'],
];
summary.getRange(`A5:C${measures.length + 4}`).values = measures;
const lengthRange = `'Cycles'!$H$5:$H$${c.last}`;
const formulas = {
  B6: `=COUNT(${lengthRange})`, B7: `=SMALL(${lengthRange},ROUNDUP(B5*B6,0))`, B8: '=B7/2', B9: '=B7+1',
  B10: `=COUNTIFS(${lengthRange},"<="&B7)/B6`, B11: `=COUNTIFS(${lengthRange},"<"&B7)/B6`,
  B13: `=MIN(${lengthRange})`, B14: `=MEDIAN(${lengthRange})`,
  B15: `=AVERAGE(${lengthRange})`, B16: `=MAX(${lengthRange})`,
};
for (const [cell, formula] of Object.entries(formulas)) summary.getRange(cell).formulas = [[formula]];
summary.getRange('A5:A28').format.columnWidth = 39;
summary.getRange('B5:B28').format.columnWidth = 24;
summary.getRange('C5:C28').format.columnWidth = 15;
summary.getRange('D4:D150').format.columnWidth = 3;
summary.getRange('E4:M150').format.columnWidth = 13;
summary.getRange('A5:C28').format.rowHeight = 21;
summary.getRange('B5').setNumberFormat('0%');
summary.getRange('B6:B21').setNumberFormat('#,##0');
summary.getRange('B8').setNumberFormat('0.0');
summary.getRange('B10:B11').setNumberFormat('0.00%');
summary.getRange('B12:B15').setNumberFormat('0.00');
summary.getRange('B22:B24').setNumberFormat('yyyy-mm-dd hh:mm');
summary.getRange('A7:C9').format.fill = '#EEF4F8';
summary.getRange('B7:B9').format.font = {name: 'Arial', size: 12, bold: true, color: ink};

const notes = [
  'Cycle = distance between consecutive green-to-red starts.',
  'Positive hist is green. Negative hist is red.',
  'Exact zeros keep the previous nonzero half-wave colour.',
  'Missing 30m slots: prior-close OHLC and zero volume.',
  'Synthetic slots are flagged in Histogram and CSV.',
  'EMA: first-value initialization, recursive adjust=False.',
  'Only completed cycles with both starts in range are used.',
];
notes.forEach((text, i) => { summary.getRange(`E${23 + i}`).values = [[text]]; });
summary.getRange('E23:M29').format.font = {name: 'Arial', size: 10, color: muted};
const dist = payload.distribution;
const lastD = 35 + dist.rows.length;
summary.getRange('F35:I35').values = [['Cycle candles','Cycle count','Cumulative share','90% reference']];
summary.getRange('F35:I35').format = {fill: header, font: {name: 'Arial', size: 10, bold: true, color: '#FFFFFF'}, wrapText: true, rowHeight: 30};
summary.getRange(`F36:I${lastD}`).values = dist.rows.map(r => [r[0],r[1],null,null]);
summary.getRange(`H36:H${lastD}`).formulas = dist.rows.map((r,i) => [`=SUM($G$36:G${36+i})/$B$6`]);
summary.getRange(`I36:I${lastD}`).formulas = dist.rows.map(() => ['=$B$5']);
summary.getRange(`F36:G${lastD}`).setNumberFormat('#,##0');
summary.getRange(`H36:I${lastD}`).setNumberFormat('0.0%');
summary.getRange(`F36:I${lastD}`).format.font = {name: 'Arial', size: 10, color: ink};
const chart = summary.charts.add('scatter', [summary.getRange(`F35:F${lastD}`),summary.getRange(`H35:H${lastD}`),summary.getRange(`I35:I${lastD}`)]);
chart.title = 'Completed cycles covered by interval length';
chart.titleTextStyle.typeface = 'Arial'; chart.titleTextStyle.fontSize = 12;
chart.legend = {position: 'top', textStyle: {typeface: 'Arial', fontSize: 10}};
chart.xAxis = {numberFormatCode:'0',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:10}};
chart.yAxis = {numberFormatCode:'0%',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:10}};
chart.xAxis.title.text = '30m periods between red starts';
chart.yAxis.title.text = 'Share of completed cycles';
chart.series.items[0].line = {fill: '#376F94', style:'solid', width:2};
chart.series.items[1].line = {fill: '#B37C37', style:'dashed', width:1};
chart.setPosition('E4','M21');

const m = addData(monthly, payload.monthly, payload.monthly.columns,
  ['Month (UTC+3)', 'Sample coverage', 'Completed cycles', 'Total 30m periods',
    'Average (30m periods)', 'Average (hours)', 'Monthly P90 (30m periods)',
    'Cycles within overall P90', 'Share within overall P90'],
  'Monthly MACD cycle lengths');
monthly.getRange('A2:A3').values = [
  ['Count each cycle once, in the month when its ending red candle closes (UTC+3).'],
  [`Overall benchmark: ${meta.mean_cycle_candles.toFixed(2)} periods on average; P90 = ${meta.p90_cycle_candles} periods. Partial months are flagged.`],
];
monthly.getRange('A2:I3').format.font = {name:'Arial', size:10, color:muted};
monthly.getRange(`A5:A${m.last}`).setNumberFormat('mmm yyyy');
monthly.getRange(`B4:B${m.last}`).format.columnWidth = 19;
monthly.getRange(`E4:E${m.last}`).format.columnWidth = 19;
monthly.getRange(`E5:F${m.last}`).setNumberFormat('0.00');
monthly.getRange(`G5:G${m.last}`).setNumberFormat('0');
monthly.getRange(`I5:I${m.last}`).setNumberFormat('0.0%');
monthly.getRange(`E5:F${m.last}`).formulasR1C1 = payload.monthly.rows.map(row =>
  row[2] ? ['=RC[-1]/RC[-2]', '=RC[-1]/2'] : [null, null]);
monthly.getRange(`I5:I${m.last}`).formulasR1C1 = payload.monthly.rows.map(row =>
  [row[2] ? '=RC[-1]/RC[-6]' : null]);
for (const [i, row] of payload.monthly.rows.entries()) {
  if (row[1] === 'Partial month') monthly.getRange(`B${i+5}`).format.font = {name:'Arial', italic:true, color:muted};
}

wb.recalculate();
console.log('Workbook formulas recalculated');
const readback = summary.getRange('B5:B16').values;
const actualP90 = readback[2][0], actualCycles = readback[1][0];
if (actualP90 !== meta.p90_cycle_candles || actualCycles !== meta.completed_cycles) {
  throw new Error(`Summary mismatch: P90=${actualP90}, cycles=${actualCycles}`);
}
const monthlyValues = monthly.getRange(`C5:I${m.last}`).values;
const totalMonthlyCycles = monthlyValues.reduce((sum, row) => sum + row[0], 0);
const totalMonthlyCovered = monthlyValues.reduce((sum, row) => sum + row[5], 0);
const monthlyWeightedMean = monthlyValues.reduce((sum, row) => sum + row[1], 0) / totalMonthlyCycles;
if (totalMonthlyCycles !== actualCycles || totalMonthlyCovered !== meta.cycles_at_or_below_p90
    || Math.abs(monthlyWeightedMean - meta.mean_cycle_candles) > 1e-10) {
  throw new Error('Monthly totals do not reconcile to the overall cycle summary');
}
for (const [i, row] of monthlyValues.entries()) {
  const expected = payload.monthly.rows[i];
  if (row[0] && [row[2] - expected[4], row[3] - expected[5], row[6] - expected[8]]
      .some(diff => !Number.isFinite(diff) || Math.abs(diff) > 1e-10)) {
    throw new Error(`Monthly formula result mismatch at row ${i + 5}`);
  }
}
console.log((await wb.inspect({kind:'table',range:'Summary!A4:C16',include:'values,formulas',tableMaxRows:13,tableMaxCols:3,maxChars:4500})).ndjson);
console.log((await wb.inspect({kind:'table',range:'Monthly!A4:I9',include:'values,formulas',tableMaxRows:6,tableMaxCols:9,maxChars:3000})).ndjson);
for (const [name, range] of [['Monthly','A1:I14']]) {
  const blob = await wb.render({sheetName:name,range,scale:1.3,format:'png'});
  await fs.writeFile(path.join(reportDir,`${name.toLowerCase()}_preview.png`),new Uint8Array(await blob.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
await fs.writeFile(path.join(reportDir,'macd_30m_report_verification.json'),JSON.stringify({
  output:outputPath, nativeCandles:meta.native_candles_in_analysis, timelineRows:h.count, completedCycles:c.count,
  p90Candles:actualP90, coverage:meta.coverage_at_p90, formulaValues:readback,
  monthlyRows:m.count, totalMonthlyCycles, totalMonthlyCovered, monthlyWeightedMean,
  sheets:['Summary','Monthly','Cycles','Histogram'], sources:[meta.source_endpoint,meta.source_documentation],
},null,2));
console.log(`Workbook saved: ${outputPath}`);
