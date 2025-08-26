# TTC Delay Analysis Webapp

## About the Project

This project was initially created as a part of the 2025 SDSS Datathon, where it was also DEMO-ed. 

However, some aspects about the project were not complete, such as the front-end, which hadn't been integrated with the predictive data. I took this break as an opportunity to modify and extend the project, and complete a front-end with . This repository is forked from the original repository for the project, and I will outline the changes and additional contributions I made, later on in this documentation.

## Set Up
Refer to [SETUP_GUIDE.md]

Everything you need to build a Svelte project, powered by [`sv`](https://github.com/sveltejs/cli).

## Creating a project

If you're seeing this, you've probably already done this step. Congrats!

```bash
# create a new project in the current directory
npx sv create

# create a new project in my-app
npx sv create my-app
```

## Developing

Once you've created a project and installed dependencies with `npm install` (or `pnpm install` or `yarn`), start a development server:

```bash
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

## Building

To create a production version of your app:

```bash
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://svelte.dev/docs/kit/adapters) for your target environment.
