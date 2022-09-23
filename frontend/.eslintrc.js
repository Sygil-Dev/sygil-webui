module.exports = {
  env: {
    node: true
  },
  'extends': [
    'plugin:vue/vue3-essential',
    "plugin:@typescript-eslint/eslint-recommended",
    "plugin:@typescript-eslint/recommended",
    'eslint:recommended',
    '@vue/typescript/recommended'
  ],
  ignorePatterns: [
    'node_modules/',
    'dist/',
    'coverage/',
    'pnpm-lock.yaml',
    '*.js'
  ],
  rules: {
    'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off'
  }
}