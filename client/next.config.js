/** @type {import('next').NextConfig} */
// const nextConfig = {
//     publicRuntimeConfig: {
//         apiUrl: process.env.NEXT_PUBLIC_API_URL,
//     },
// }

// module.exports = nextConfig

module.exports = {
    publicRuntimeConfig: {
        apiUrl: process.env.NEXT_PUBLIC_API_URL,
    },
}
