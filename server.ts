import { readFileSync } from 'fs';
import { join } from 'path';

const server = Bun.serve({
    port: 3000,
    async fetch(req) {
        const url = new URL(req.url);
        const path = url.pathname;

        // Serve index.html for root
        if (path === '/' || path === '/index.html') {
            return new Response(readFileSync(join(import.meta.dir, 'index.html')), {
                headers: { 'Content-Type': 'text/html' }
            });
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
                        headers: { 'Content-Type': 'application/javascript' }
                    });
                }
            } catch (e) {
                console.error('Build error:', e);
                return new Response('Build error', { status: 500 });
            }
        }

        return new Response('Not found', { status: 404 });
    }
});

console.log(`Server running at http://localhost:${server.port}`);
