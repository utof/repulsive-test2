import { describe, expect, test } from 'bun:test';

import { classifyAdapter } from '../../bench/gpu/adapterClass';

/**
 * Gate provenance depends on this predicate: spec §3 rules that any result
 * obtained against a software rasterizer is INVALID, never a passing (or
 * failing) gate measurement.
 *
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3
 * @issue utof/repulsive-test2#43
 */
describe('classifyAdapter', () => {
    test('identifies SwiftShader by description', () => {
        expect(
            classifyAdapter({ vendor: 'Google Inc.', description: 'SwiftShader Device (LLVM)' }),
        ).toBe('software');
    });

    test('identifies llvmpipe by vendor', () => {
        expect(classifyAdapter({ vendor: 'llvmpipe', description: 'Mesa' })).toBe('software');
    });

    test('identifies a bare "software" marker', () => {
        expect(classifyAdapter({ vendor: 'Mesa', description: 'software rasterizer' })).toBe(
            'software',
        );
    });

    test('matches case-insensitively', () => {
        expect(classifyAdapter({ vendor: 'LLVMPIPE', description: '' })).toBe('software');
    });

    test('classifies a real hardware adapter as hardware', () => {
        expect(classifyAdapter({ vendor: 'NVIDIA', description: 'Quadro RTX 3000 (Turing)' })).toBe(
            'hardware',
        );
    });

    // The regression this module exists for. Firefox blanks `adapterInfo`, and
    // empty strings match none of the software markers — so the pre-fix
    // substring test fell through to 'hardware' and a software-fallback run
    // would have been recorded as a valid hardware gate measurement.
    // `drive.ts` already rules this way for the undeterminable-adapter path
    // ("genuinely unknown, treated as 'software' so it never masquerades as a
    // passing hardware run"); these cases close the same hole for a blank one.
    // @issue utof/repulsive-test2#51
    test('treats blank adapter info as software, not hardware', () => {
        expect(classifyAdapter({ vendor: '', description: '' })).toBe('software');
    });

    test('treats whitespace-only adapter info as software', () => {
        expect(classifyAdapter({ vendor: '   ', description: '\t\n' })).toBe('software');
    });

    test('treats missing fields as software', () => {
        expect(classifyAdapter({} as { vendor: string; description: string })).toBe('software');
        expect(
            classifyAdapter({ vendor: undefined, description: undefined } as unknown as {
                vendor: string;
                description: string;
            }),
        ).toBe('software');
    });

    // The second defect on this branch: the #24 four-field fix was never
    // forward-ported here, so `architecture` and `device` went unread. Chrome's
    // SwiftShader adapter leaves `description` empty and puts its only matchable
    // token in `architecture` — a non-blank record that the blank guard above
    // does NOT catch, and that vendor+description alone calls 'hardware'.
    // @issue utof/repulsive-test2#24
    test('identifies SwiftShader when only architecture carries the marker', () => {
        expect(
            classifyAdapter({
                vendor: 'Google Inc. (Google)',
                architecture: 'swiftshader',
                device: '',
                description: '',
            }),
        ).toBe('software');
    });

    test('identifies a software marker carried only by device', () => {
        expect(
            classifyAdapter({
                vendor: 'Mesa',
                architecture: '',
                device: 'llvmpipe (LLVM 15.0.7, 256 bits)',
                description: '',
            }),
        ).toBe('software');
    });

    test('reads all four fields for a real hardware adapter', () => {
        expect(
            classifyAdapter({
                vendor: 'nvidia',
                architecture: 'turing',
                device: '',
                description: '',
            }),
        ).toBe('hardware');
    });

    test('a vendor alone is enough to certify hardware', () => {
        expect(classifyAdapter({ vendor: 'NVIDIA', description: '' })).toBe('hardware');
    });
});
