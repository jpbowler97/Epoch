# Claude Code Benchmark Update Task

## Objective
Update benchmark data from JavaScript-rendered websites that regular Python scrapers cannot access. You will use your WebFetch tool to get the rendered content and save it as properly formatted HTML files.

## Context
These benchmark sites use dynamic JavaScript rendering (Gradio, React, etc.) that makes them inaccessible to standard web scraping tools. Your WebFetch capability can handle these sites by rendering the JavaScript before extraction.

## Sites to Update

{{ sites_to_update }}

## General Requirements

1. **HTML Structure**: Each file must be a valid HTML document with:
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <meta charset="UTF-8">
       <title>{{ site_name }} Benchmark Data</title>
       <!-- Updated: {{ timestamp }} -->
   </head>
   <body>
       <table>
           <thead>
               <tr>
                   <th>Column1</th>
                   <th>Column2</th>
                   <!-- etc -->
               </tr>
           </thead>
           <tbody>
               <!-- Data rows -->
           </tbody>
       </table>
   </body>
   </html>
   ```

2. **Data Preservation**:
   - Keep all model names exactly as found
   - Preserve numeric scores with original precision
   - Maintain any special characters or Unicode text
   - Include all available columns even if not explicitly requested

3. **Error Handling**:
   - If a site is unreachable, create a minimal HTML file with an error comment
   - If data is partial, include what you can extract
   - Log any issues encountered but continue with other sites

4. **Validation**:
   - Ensure each table has at least 5 rows of data (or document if fewer exist)
   - Verify column headers are present and meaningful
   - Check that numeric scores are properly formatted

## Post-Processing
After creating all files, briefly report:
- Number of models found for each site
- Any issues encountered
- Confirmation that files were saved to specified paths

## Important Notes
- Use WebFetch's ability to handle JavaScript-rendered content
- Some sites may require waiting for dynamic content to load
- Extract the most complete and recent data available
- If multiple tables exist, choose the main leaderboard/ranking table