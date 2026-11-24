// bench/app/driveApp.ts — bun bench/app/driveApp.ts run:<vertices>:<steps>[:<driver>]
//   also: prepass:<nodes>:<edges>:<steps> | untangle:<vertices>:<steps> | graphflow:<size>:<ms>
// Opens the real app in a headless browser (the same host layer the GPU gates use), presses
// play on the large trefoil preset, and prints what the frame loop did as one JSON line.
// Needs the dev server: `PORT=3111 bun server.ts`, then DEV_SERVER_URL=http://localhost:3111.
// This is a does-it-run check of the CPU step inside a browser, not a GPU gate: the adapter is
// not certified and no GPU number is read from it.
// @see bench/app/appHarness.tsx, bench/gpu/drive.ts
import { selectHost } from '../gpu/hosts';

const spike = process.argv[2];
if (!spike) {
    console.error('usage: bun bench/app/driveApp.ts run:<vertices>:<steps>[:<driver>]');
    process.exit(2);
}
const PORT = Number(process.env.GPU_HOST_PORT ?? 9224);
const DEV_SERVER = process.env.DEV_SERVER_URL ?? 'http://localhost:3000';

const host = selectHost(process.env.GPU_HOST);
const variant = host.resolveVariant(host.launchVariants[0]);
console.error(`[driveApp] host=${host.id} (${host.version()}) variant=${variant.label}`);
const session = await host.launch(variant, `${DEV_SERVER}/bench/app/appHarness.html`, PORT);
try {
    const data = await session.evalSpike(spike);
    console.log(JSON.stringify(data));
    const noise = session.consoleLines.filter((l) => /error|solverDriver|dispatch/i.test(l));
    if (noise.length)
        console.error(`[driveApp] console (${noise.length}):\n${noise.slice(0, 12).join('\n')}`);
} finally {
    session.close();
}
