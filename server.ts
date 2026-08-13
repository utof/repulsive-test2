import { readFileSync } from 'fs';
import { join } from 'path';

const port = parseInt(process.env.PORT || '3000');

const server = Bun.serve({
    port,
    async fetch(req) {
        const url = new URL(req.url);
        const path = url.pathname;

        // Serve index.html for root
        if (path === '/' || path === '/index.html') {
            return new Response(readFileSync(join(import.meta.dir, 'index.html')), {
                headers: { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' },
            });
        }

        // Serve any other static .html file (e.g. bench/gpu/harness.html) — mirrors
        // the index.html handler above. @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 1
        if (path.endsWith('.html')) {
            const filePath = join(import.meta.dir, path);
            try {
                return new Response(readFileSync(filePath), {
                    headers: { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' },
                });
            } catch {
                return new Response('Not found', { status: 404 });
            }
        }

        // Bundle and serve TypeScript/TSX files
        if (path.endsWith('.tsx') || path.endsWith('.ts')) {
            const filePath = join(import.meta.dir, path);
            try {
                const result = await Bun.build({
                    entrypoints: [filePath],
                    target: 'browser',
                    minify: false,
                });

                if (result.success && result.outputs.length > 0) {
                    const code = await result.outputs[0].text();
                    return new Response(code, {
                        headers: {
                            'Content-Type': 'application/javascript',
                            // Dev server rebuilds per request; without this the browser can serve a
                            // heuristically-cached module bundle, so source edits appear to do nothing.
                            'Cache-Control': 'no-store',
                        },
                    });
                }
            } catch (e) {
                console.error('Build error:', e);
                return new Response('Build error', { status: 500 });
            }
        }

        return new Response('Not found', { status: 404 });
    },
});

console.log(`Server running at http://localhost:${server.port}`);
